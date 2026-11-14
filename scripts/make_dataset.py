"""고정 데이터셋 생성: train / pool / test_normal / test_drift 를 data/dataset/ 아래에 한 번에 만든다.

실행: python -m scripts.make_dataset [--regenerate]

- Collector 로 만들고, 세트마다 시드를 고정해서 같은 명령이면 같은 이미지가 나온다.
- 세트 사이에 같은 이미지(내용 해시)가 있거나 inspection_id 가 겹치면 실패하고 만든 것을 지운다.
- 이미 만들어져 있으면 덮어쓰지 않고 안내만 한다. --regenerate 를 줄 때만 data/dataset 을 지우고 다시 만든다.
- 이미지마다 정보(라벨, 미체결 번호, 드리프트 종류 등)는 manifest.jsonl 에, 요약은 dataset_info.json 에 쓴다.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from common.schemas import Inspection, validate_pil_image
from parts.collector import Collector
from parts.collector.config import CollectorConfig, load_config
from parts.collector.simulator import DRIFT_TYPES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = "data/dataset"  # 프로젝트 루트 기준. 운영 중 수집이 쓰는 data/pool/ 등과 분리한다
MANIFEST = "manifest.jsonl"
INFO = "dataset_info.json"
FIXED_TIME = datetime(2026, 10, 19, 9, 0, 0, tzinfo=timezone(timedelta(hours=9)))  # 같은 명령이면 같은 파일이 나오게 고정


class DatasetError(RuntimeError):
    """데이터셋 생성·검사 실패."""


class RefusedPathError(DatasetError):
    """지우려는 경로가 프로젝트의 data/dataset 이 아니어서 거부함."""


@dataclass(frozen=True)
class SplitSpec:
    split: str
    size: int
    scenario: str  # normal 또는 drift
    seed: int


# 문서에 있는 값: test_normal 500, test_drift 500 (정상 400 + 불량 100).
# 가정: train 2,000, pool 500 (스파이크가 쓴 건수). 불량은 모두 정확히 20%.
# 만드는 순서(= inspection_id 가 이어지는 순서)는 train, pool, test_normal, test_drift.
DEFAULT_SPECS = (
    SplitSpec("train", 2000, "normal", 1001),
    SplitSpec("pool", 500, "drift", 1004),
    SplitSpec("test_normal", 500, "normal", 1002),
    SplitSpec("test_drift", 500, "drift", 1003),
)


# ---------- 검사 ----------

def check_unique_ids(rows: list) -> None:
    ids = [r["inspection_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise DatasetError(f"inspection_id 가 겹칩니다 (전체 {len(ids)}개 중 서로 다른 값 {len(set(ids))}개)")


def check_no_overlap(hashes_by_split: dict) -> None:
    """세트 사이에 같은 이미지(내용 해시)가 하나도 없어야 한다."""
    names = list(hashes_by_split)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = hashes_by_split[a] & hashes_by_split[b]
            if common:
                raise DatasetError(f"{a} 와 {b} 에 같은 이미지가 {len(common)}장 있습니다")


def remove_dataset(root, path, log=print) -> bool:
    """삭제 대상이 프로젝트의 data/dataset 과 정확히 같을 때만 지운다. 그 밖의 경로는 거부한다."""
    allowed = os.path.normcase(str((Path(root) / DATASET_DIR).resolve()))
    path = Path(path)
    if path.is_symlink() or os.path.normcase(str(path.resolve())) != allowed:
        raise RefusedPathError(f"삭제를 거부합니다. 지울 수 있는 경로는 {allowed} 하나뿐입니다 (받은 경로: {path})")
    if not path.is_dir():
        return False
    log(f"삭제: {path}")
    shutil.rmtree(path)
    return True


# ---------- 생성 ----------

def _ng_count(size: int, defect_rate: float) -> int:
    return round(size * defect_rate)


def _plan(spec: SplitSpec, defect_rate: float) -> list:
    """(라벨, 드리프트 종류) 목록. 불량 개수를 정확히 맞추고, 드리프트 종류는 고르게 돌린다."""
    n_ng = _ng_count(spec.size, defect_rate)
    labels = ["NG"] * n_ng + ["OK"] * (spec.size - n_ng)
    if spec.scenario == "drift":
        kinds = [DRIFT_TYPES[i % len(DRIFT_TYPES)] for i in range(spec.size)]  # 불량부터 이어서 돌려 종류별 불량도 고름
    else:
        kinds = [None] * spec.size
    items = list(zip(labels, kinds))
    order = np.random.default_rng(spec.seed).permutation(spec.size)
    return [items[i] for i in order]


def _write_collector_config(path: Path, cfg: CollectorConfig) -> None:
    """collector.toml 과 같되 data_dir 만 data/dataset 인 설정 (parts/ 는 수정하지 않는다)."""
    lines = [
        f"defect_rate = {cfg.defect_rate!r}",
        f"image_size = {cfg.image_size}",
        f"part_count = {cfg.part_count}",
        f"seed = {cfg.seed}",
        f'data_dir = "{DATASET_DIR}"',
        "[drift_strength]",
        *[f"{k} = {v!r}" for k, v in cfg.drift_strength.items()],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generate(root: Path, specs, cfg: CollectorConfig, log) -> list:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "collector.toml"
        _write_collector_config(config_path, cfg)
        for spec in specs:
            collector = Collector(config_path=config_path, root=root, seed=spec.seed, clock=lambda: FIXED_TIME)
            for label, kind in _plan(spec, cfg.defect_rate):
                sample = collector.next_sample(
                    scenario=spec.scenario, data_split=spec.split, drift_type=kind, label=label)
                rows.append({**sample.inspection.to_dict(),
                             "image_md5": hashlib.md5(sample.image.tobytes()).hexdigest()})
            log(f"{spec.split}: {spec.size}장 생성")
    return rows


def _summarize(rows: list, specs, defect_rate: float) -> dict:
    """세트별 건수·불량 건수·종류별 건수를 확인하고 요약을 만든다. 어긋나면 DatasetError."""
    splits = {}
    for spec in specs:
        part = [r for r in rows if r["data_split"] == spec.split]
        ng = sum(r["true_label"] == "NG" for r in part)
        kinds = {k: sum(r["drift_type"] == k for r in part) for k in DRIFT_TYPES} if spec.scenario == "drift" else {}
        if len(part) != spec.size or ng != _ng_count(spec.size, defect_rate):
            raise DatasetError(f"{spec.split}: 건수가 맞지 않습니다 (이미지 {len(part)}장, 불량 {ng}장)")
        if any(r["scenario"] != spec.scenario for r in part):
            raise DatasetError(f"{spec.split}: 시나리오가 {spec.scenario} 가 아닌 이미지가 있습니다")
        if kinds and max(kinds.values()) - min(kinds.values()) > 1:
            raise DatasetError(f"{spec.split}: 드리프트 종류가 고르지 않습니다 ({kinds})")
        if len({r["image_md5"] for r in part}) != len(part):
            raise DatasetError(f"{spec.split}: 같은 세트 안에 같은 이미지가 있습니다")
        splits[spec.split] = {"size": spec.size, "ng": ng, "ok": spec.size - ng, "scenario": spec.scenario,
                              "seed": spec.seed, "drift_types": kinds}
    check_unique_ids(rows)
    check_no_overlap({s: {r["image_md5"] for r in rows if r["data_split"] == s} for s in splits})
    return {"total": len(rows), "splits": splits, "image_overlap_between_splits": 0, "inspection_ids_unique": True,
            "collector": {"defect_rate": defect_rate, "drift_strength": load_config().drift_strength},
            "captured_at_fixed": FIXED_TIME.isoformat()}


def _is_complete(target: Path) -> bool:
    return (target / MANIFEST).is_file() and (target / INFO).is_file()


def build_dataset(root=PROJECT_ROOT, specs=DEFAULT_SPECS, regenerate: bool = False, log=print) -> dict:
    """데이터셋을 만든다. 이미 있으면 {'status': 'exists'}, 새로 만들면 {'status': 'created', 'info': ...}."""
    root = Path(root)
    target = root / DATASET_DIR
    names = [s.split for s in specs]
    if len(set(names)) != len(names) or not set(names) <= {"train", "pool", "test_normal", "test_drift"}:
        raise ValueError(f"specs: 세트 이름은 train/pool/test_normal/test_drift 를 한 번씩만 쓸 수 있습니다 ({names})")

    if target.exists():
        if not regenerate:
            if _is_complete(target):
                log(f"이미 만들어져 있습니다: {target} (다시 만들려면 --regenerate)")
                return {"status": "exists", "path": str(target)}
            raise DatasetError(f"불완전한 데이터셋이 있습니다: {target}. 지우고 다시 만들려면 --regenerate 를 쓰세요")
        remove_dataset(root, target, log)

    cfg = load_config()
    try:
        rows = _generate(root, specs, cfg, log)
        info = _summarize(rows, specs, cfg.defect_rate)
        with open(target / MANIFEST, "w", encoding="utf-8", newline="\n") as f:  # 설명 파일은 마지막에 쓴다
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (target / INFO).write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        remove_dataset(root, target, log=lambda *_: None)  # 반쪽짜리 데이터셋을 남기지 않는다
        raise
    return {"status": "created", "path": str(target), "info": info}


# ---------- 읽기 (measure_baseline 이 사용) ----------

@dataclass
class SplitData:
    images: np.ndarray  # uint8 (N, 128, 128)
    labels: list        # "OK" / "NG"
    kinds: list         # 드리프트 종류 (정상 환경이면 None)
    ids: list


def load_dataset(root=PROJECT_ROOT):
    """manifest.jsonl 을 읽어 세트별 이미지·라벨·종류를 돌려준다. 이미지는 해시까지 다시 확인한다."""
    root = Path(root)
    target = root / DATASET_DIR
    if not _is_complete(target):
        raise DatasetError(f"데이터셋이 없습니다: {target}. 먼저 python -m scripts.make_dataset 을 실행하세요")
    images, meta = {}, {}
    for line in (target / MANIFEST).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        md5 = row.pop("image_md5")
        inspection = Inspection.from_dict(row)
        with Image.open(root / inspection.image_path) as png:
            png.load()
            validate_pil_image(png)
            arr = np.asarray(png).copy()
        if hashlib.md5(arr.tobytes()).hexdigest() != md5:
            raise DatasetError(f"{inspection.image_path}: 이미지가 만들 때와 다릅니다 (해시 불일치)")
        split = inspection.data_split.value
        images.setdefault(split, []).append(arr)
        meta.setdefault(split, []).append(inspection)
    data = {s: SplitData(np.stack(images[s]), [i.true_label.value for i in meta[s]],
                         [i.drift_type for i in meta[s]], [i.inspection_id for i in meta[s]]) for s in images}
    return data, json.loads((target / INFO).read_text(encoding="utf-8"))


def main(argv=None, root=PROJECT_ROOT) -> int:
    parser = argparse.ArgumentParser(description="고정 데이터셋(train/pool/test_normal/test_drift)을 data/dataset 에 만든다")
    parser.add_argument("--regenerate", action="store_true", help="이미 있으면 data/dataset 을 지우고 다시 만든다")
    args = parser.parse_args(argv)
    try:
        result = build_dataset(root, regenerate=args.regenerate)
    except DatasetError as e:
        print(f"실패: {e}", file=sys.stderr)
        return 1
    if result["status"] == "created":
        info = result["info"]
        print(f"만든 곳: {result['path']}")
        print(f"전체: {info['total']}장")
        for name, s in info["splits"].items():
            kinds = f", 종류 {s['drift_types']}" if s["drift_types"] else ""
            print(f"{name}: {s['size']}장 (정상 {s['ok']}, 불량 {s['ng']}{kinds}, 시드 {s['seed']})")
        print(f"세트 간 겹치는 이미지: {info['image_overlap_between_splits']}장")
        print(f"inspection_id 고유: {info['inspection_ids_unique']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
