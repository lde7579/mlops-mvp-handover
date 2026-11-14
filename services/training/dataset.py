"""재학습에 쓰는 고정 데이터셋 읽기: data/dataset 의 manifest.jsonl 에서 train, test_normal, test_drift 를 읽는다.

- DB, torch 를 쓰지 않는다 (numpy, Pillow 뿐). 이미지 → 배열 변환은 백엔드 담당이다 (계약 C3).
- data/dataset/pool 은 오프라인 측정(scripts.measure_baseline)용이라 여기서 읽지 않는다. 운영 중 재학습의 신규 데이터는 DB 의 pool 이다.
- 이미지는 만들 때 기록한 해시(image_md5)와 다시 대조한다. 다르면 손상된 것이므로 읽기를 멈춘다.
- scripts/make_dataset.py 를 가져다 쓰지 않는다 (서버 코드가 scripts/ 에 기대지 않게, 그리고 필요한 세트만 읽기 위해).
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from common.schemas import Inspection, SchemaError, validate_pil_image

DATASET_DIR = "data/dataset"  # 프로젝트 루트 기준
MANIFEST = "manifest.jsonl"
INFO = "dataset_info.json"
FIXED_SPLITS = ("train", "test_normal", "test_drift")


class TrainingDataError(RuntimeError):
    """학습 데이터를 읽을 수 없다 (데이터셋이 없거나, 이미지가 손상되었거나, 규격이 맞지 않음)."""


@dataclass
class LabeledImages:
    images: np.ndarray  # uint8 (N, 128, 128)
    labels: list        # "OK" / "NG"


def dataset_available(root) -> bool:
    folder = Path(root) / DATASET_DIR
    return (folder / MANIFEST).is_file() and (folder / INFO).is_file()


def read_png(path) -> np.ndarray:
    """PNG 한 장을 uint8 (128, 128) 배열로 읽는다. 열 수 없거나 규격이 맞지 않으면 TrainingDataError."""
    try:
        with Image.open(path) as png:
            png.load()
            validate_pil_image(png)
            return np.asarray(png).copy()
    except (OSError, SchemaError) as e:
        raise TrainingDataError(f"이미지를 읽을 수 없습니다 ({path}): {e}") from e


def stack(arrays: list, labels: list) -> LabeledImages:
    if not arrays:
        raise TrainingDataError("이미지가 하나도 없습니다")
    return LabeledImages(np.stack(arrays), list(labels))


def load_fixed_sets(root) -> dict:
    """{'train': LabeledImages, 'test_normal': ..., 'test_drift': ...}. 세트가 비어 있으면 TrainingDataError."""
    root = Path(root)
    if not dataset_available(root):
        raise TrainingDataError(f"고정 데이터셋이 없습니다: {root / DATASET_DIR}")
    arrays = {s: [] for s in FIXED_SPLITS}
    labels = {s: [] for s in FIXED_SPLITS}
    for line in (root / DATASET_DIR / MANIFEST).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("data_split") not in FIXED_SPLITS:
            continue  # pool 등은 읽지 않는다
        md5 = row.pop("image_md5", None)
        try:
            inspection = Inspection.from_dict(row)
        except SchemaError as e:
            raise TrainingDataError(f"manifest 행이 계약에 맞지 않습니다: {e}") from e
        image = read_png(root / inspection.image_path)
        if md5 is not None and hashlib.md5(image.tobytes()).hexdigest() != md5:
            raise TrainingDataError(f"{inspection.image_path}: 이미지가 만들 때와 다릅니다 (해시 불일치)")
        split = inspection.data_split.value
        arrays[split].append(image)
        labels[split].append(inspection.true_label.value)
    return {s: stack(arrays[s], labels[s]) for s in FIXED_SPLITS}
