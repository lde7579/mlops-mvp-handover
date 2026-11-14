"""services/training/dataset.py: 고정 데이터셋(train, test_normal, test_drift) 읽기. torch·DB 없이 돈다."""
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from services.training.dataset import FIXED_SPLITS, TrainingDataError, dataset_available, load_fixed_sets, read_png
from tests.backend.helpers import make_inspection

SPLITS = [("train", "OK", 1), ("train", "NG", 2), ("pool", "NG", 3), ("test_normal", "OK", 4), ("test_drift", "NG", 5)]


def build(root, *, corrupt_hash=False):
    """작은 data/dataset 을 만든다 (train 2장, pool 1장, test_normal 1장, test_drift 1장)."""
    folder = root / "data" / "dataset"
    rows = []
    for split, label, n in SPLITS:
        (folder / split).mkdir(parents=True, exist_ok=True)
        image = np.full((128, 128), n * 20, dtype=np.uint8)
        path = f"data/dataset/{split}/INS-{n:06d}.png"
        Image.fromarray(image).save(root / path)
        scenario = "drift" if split in ("test_drift", "pool") else "normal"
        inspection = make_inspection(n, true_label=label, missing_parts=[2] if label == "NG" else [], image_path=path,
                                     data_split=split, scenario=scenario, drift_type="dark" if scenario == "drift" else None)
        md5 = "0" * 32 if corrupt_hash and n == 1 else hashlib.md5(image.tobytes()).hexdigest()
        rows.append({**inspection.to_dict(), "image_md5": md5})
    (folder / "manifest.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (folder / "dataset_info.json").write_text("{}", encoding="utf-8")


def test_loads_only_the_fixed_sets_and_skips_the_dataset_pool(tmp_path):
    build(tmp_path)
    data = load_fixed_sets(tmp_path)
    assert set(data) == set(FIXED_SPLITS) and "pool" not in data
    assert data["train"].images.shape == (2, 128, 128) and data["train"].images.dtype == np.uint8
    assert data["train"].labels == ["OK", "NG"] and data["test_normal"].labels == ["OK"] and data["test_drift"].labels == ["NG"]
    assert int(data["train"].images[1][0, 0]) == 40  # manifest 순서 그대로


def test_availability_needs_manifest_and_info(tmp_path):
    assert not dataset_available(tmp_path)
    build(tmp_path)
    assert dataset_available(tmp_path)
    (tmp_path / "data" / "dataset" / "dataset_info.json").unlink()
    assert not dataset_available(tmp_path)


def test_missing_dataset_is_a_training_data_error(tmp_path):
    with pytest.raises(TrainingDataError):
        load_fixed_sets(tmp_path)


def test_hash_mismatch_and_missing_image_are_refused(tmp_path):
    build(tmp_path, corrupt_hash=True)
    with pytest.raises(TrainingDataError, match="해시"):
        load_fixed_sets(tmp_path)
    build(tmp_path)
    (tmp_path / "data" / "dataset" / "train" / "INS-000002.png").unlink()
    with pytest.raises(TrainingDataError):
        load_fixed_sets(tmp_path)


def test_wrong_image_format_is_refused(tmp_path):
    build(tmp_path)
    Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(tmp_path / "data" / "dataset" / "train" / "INS-000001.png")
    with pytest.raises(TrainingDataError):
        read_png(tmp_path / "data" / "dataset" / "train" / "INS-000001.png")
    with pytest.raises(TrainingDataError):
        load_fixed_sets(tmp_path)


def test_empty_fixed_set_is_refused(tmp_path):
    build(tmp_path)
    manifest = tmp_path / "data" / "dataset" / "manifest.jsonl"
    kept = [line for line in manifest.read_text(encoding="utf-8").splitlines() if '"data_split": "test_drift"' not in line]
    manifest.write_text("\n".join(kept) + "\n", encoding="utf-8")
    with pytest.raises(TrainingDataError):
        load_fixed_sets(tmp_path)
