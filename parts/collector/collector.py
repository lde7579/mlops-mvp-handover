"""수집기: 검사 1건(이미지 + Inspection 메타데이터)을 만들어 PNG로 저장한다.

저장 전 검증(계약 2장): 이미지 128x128 흑백, OK이면 missing_parts 비어 있음, 파일이 열리는지,
ID 중복 없음. 하나라도 어긋나면 파일을 남기지 않고 오류를 낸다.
missing_parts 는 "미체결 볼트 번호(1~6)"로 해석한다.
"""
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from common.schemas import (
    DataSplit,
    Inspection,
    Label,
    Scenario,
    SchemaError,
    validate_image_array,
    validate_pil_image,
)
from parts.collector.config import load_config
from parts.collector.simulator import BOLT_COUNT, DRIFT_TYPES, make_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = tuple(s.value for s in DataSplit)
_ID_FILE = re.compile(r"^INS-(\d+)\.png$")
_log = logging.getLogger("collector")


@dataclass
class Sample:
    inspection: Inspection
    image: np.ndarray  # uint8, (128, 128)


def _parse(enum_cls, name, value):
    try:
        return enum_cls(value)
    except ValueError:
        allowed = ", ".join(m.value for m in enum_cls)
        raise SchemaError(f"{name}: 허용값은 {allowed} 입니다 (받은 값: {value!r})") from None


class Collector:
    """한 번에 하나의 Collector 만 같은 data 폴더에 쓴다고 가정한다."""

    def __init__(self, config_path=None, *, root=None, seed=None, clock=None):
        self.config = load_config(config_path)
        self.root = Path(root) if root is not None else PROJECT_ROOT
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError(f"seed: 정수여야 합니다 (받은 값: {seed!r})")
        self._rng = np.random.default_rng(self.config.seed if seed is None else seed)
        self._clock = clock or (lambda: datetime.now().astimezone())
        self._data_dir = self.root / self.config.data_dir
        self._next_id = self._last_saved_id() + 1

    def _last_saved_id(self) -> int:
        """data/<data_split>/ 의 가장 큰 번호. data/samples 등 다른 폴더는 보지 않는다."""
        last = 0
        for split in SPLITS:
            folder = self._data_dir / split
            if folder.is_dir():
                for p in folder.iterdir():
                    m = _ID_FILE.match(p.name)
                    if m:
                        last = max(last, int(m.group(1)))
        return last

    def next_sample(self, *, scenario="normal", data_split="pool", drift_type=None, label=None) -> Sample:
        scenario = _parse(Scenario, "scenario", scenario)
        data_split = _parse(DataSplit, "data_split", data_split)
        if label is not None:
            label = _parse(Label, "label", label)

        if scenario is Scenario.NORMAL:
            if drift_type is not None:
                raise ValueError("drift_type: scenario 가 normal 이면 지정할 수 없습니다")
        elif drift_type is None:
            drift_type = DRIFT_TYPES[int(self._rng.integers(len(DRIFT_TYPES)))]
        elif drift_type not in DRIFT_TYPES:
            raise ValueError(f"drift_type: 허용값은 {', '.join(DRIFT_TYPES)} 입니다 (받은 값: {drift_type!r})")

        if label is None:
            label = Label.NG if self._rng.random() < self.config.defect_rate else Label.OK
        unfastened = []
        if label is Label.NG:
            k = int(self._rng.integers(1, 3))  # 1~2개 미체결
            unfastened = sorted(int(m) + 1 for m in self._rng.choice(BOLT_COUNT, size=k, replace=False))

        strength = self.config.drift_strength.get(drift_type, 0.0)
        image = make_image(self._rng, unfastened, drift_type, strength)

        inspection_id = f"INS-{self._next_id:06d}"
        rel_path = f"{self.config.data_dir}/{data_split.value}/{inspection_id}.png"
        path = self.root / rel_path
        try:
            validate_image_array(image)
            inspection = Inspection(
                inspection_id=inspection_id,
                captured_at=self._clock(),
                image_path=rel_path,
                true_label=label,
                missing_parts=unfastened,
                scenario=scenario,
                drift_type=drift_type,
                data_split=data_split,
            )
            self._save_png(inspection_id, path, image)
        except Exception as e:
            _log.error("수집 실패 (%s): %s", inspection_id, e)
            raise
        self._next_id += 1
        return Sample(inspection=inspection, image=image)

    def _save_png(self, inspection_id: str, path: Path, image: np.ndarray) -> None:
        for split in SPLITS:  # ID 는 모든 폴더에서 고유해야 한다
            if (self._data_dir / split / f"{inspection_id}.png").exists():
                raise FileExistsError(f"이미 있는 ID 입니다: {inspection_id}")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "xb") as f:
                Image.fromarray(image).save(f, format="PNG")
            with Image.open(path) as saved:  # 파일이 열리고 규칙에 맞는지 다시 확인
                saved.load()
                validate_pil_image(saved)
        except Exception:
            path.unlink(missing_ok=True)
            raise
