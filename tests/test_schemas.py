"""common/schemas.py 테스트 — docs/data-contract.md v1.0 의 2~4장, 1장 이미지 규칙."""
import copy

import numpy as np
import pytest
from PIL import Image

from common import preprocess
from common.schemas import (
    IMAGE_SIZE,
    ClassMetrics,
    DataSplit,
    InferenceOutput,
    Inspection,
    Label,
    ModelMetrics,
    Prediction,
    SchemaError,
    validate_image_array,
    validate_pil_image,
)


def inspection_dict(**over):
    d = {
        "inspection_id": "INS-000001",
        "captured_at": "2026-10-17T21:30:00+09:00",
        "image_path": "data/train/INS-000001.png",
        "true_label": "NG",
        "missing_parts": [3],
        "scenario": "normal",
        "data_split": "train",
    }
    d.update(over)
    return d


def prediction_dict(**over):
    d = {
        "prediction_id": "PRD-000001",
        "inspection_id": "INS-000001",
        "model_version": "v1",
        "predicted_label": "NG",
        "score": 0.9,
        "threshold": 0.5,
        "latency_ms": 12,
        "predicted_at": "2026-10-17T21:30:01+09:00",
    }
    d.update(over)
    return d


def class_metrics_dict(**over):
    d = {
        "accuracy": 1.0, "recall": 1.0, "precision": 1.0, "f1": 1.0,
        "confusion_matrix": [[400, 0], [0, 100]], "n": 500,
    }
    d.update(over)
    return d


def metrics_dict(**over):
    d = {
        "version": "v1",
        "threshold": 0.5,
        "normal": class_metrics_dict(),
        "drift": class_metrics_dict(accuracy=0.8, confusion_matrix=[[300, 100], [0, 100]]),
        "train_data_snapshot": {"train": 2000, "pool": 0},
        "created_at": "2026-10-17T22:00:00+09:00",
    }
    d.update(over)
    return d


# ---------- inspection ----------

def test_inspection_valid_ng_and_ok():
    ng = Inspection.from_dict(inspection_dict())
    assert ng.true_label is Label.NG and ng.missing_parts == [3]
    ok = Inspection.from_dict(inspection_dict(true_label="OK", missing_parts=[]))
    assert ok.true_label is Label.OK


def test_inspection_round_trip():
    d = inspection_dict(scenario="drift", drift_type="dark", data_split="test_drift")
    obj = Inspection.from_dict(d)
    assert obj.to_dict() == d
    assert Inspection.from_dict(obj.to_dict()) == obj


def test_inspection_drift_type_optional_and_open():
    assert Inspection.from_dict(inspection_dict()).drift_type is None
    # 문서는 "dark / noise / shift 등" 이라 값을 제한하지 않는다
    assert Inspection.from_dict(inspection_dict(scenario="drift", drift_type="blur")).drift_type == "blur"


@pytest.mark.parametrize("bad", ["ok", "BAD", "", None, 1])
def test_inspection_rejects_bad_label(bad):
    with pytest.raises(SchemaError):
        Inspection.from_dict(inspection_dict(true_label=bad))


@pytest.mark.parametrize("good", ["train", "pool", "test_normal", "test_drift"])
def test_inspection_accepts_data_split_values(good):
    assert Inspection.from_dict(inspection_dict(data_split=good)).data_split == DataSplit(good)


@pytest.mark.parametrize("bad", ["validation", "test", "TRAIN", "", None])
def test_inspection_rejects_bad_data_split(bad):
    with pytest.raises(SchemaError):
        Inspection.from_dict(inspection_dict(data_split=bad))


@pytest.mark.parametrize("bad", ["Normal", "other", "", None])
def test_inspection_rejects_bad_scenario(bad):
    with pytest.raises(SchemaError):
        Inspection.from_dict(inspection_dict(scenario=bad))


@pytest.mark.parametrize(
    "field",
    ["inspection_id", "captured_at", "image_path", "true_label", "missing_parts", "scenario", "data_split"],
)
def test_inspection_required_fields(field):
    d = inspection_dict()
    del d[field]
    with pytest.raises(SchemaError):
        Inspection.from_dict(d)


def test_inspection_rejects_unknown_field():
    with pytest.raises(SchemaError):
        Inspection.from_dict(inspection_dict(data_spilt="train"))


def test_inspection_ok_requires_empty_missing_parts():
    with pytest.raises(SchemaError):
        Inspection.from_dict(inspection_dict(true_label="OK", missing_parts=[2]))


@pytest.mark.parametrize("bad", ["3", [1.5], [True], None])
def test_inspection_rejects_bad_missing_parts(bad):
    with pytest.raises(SchemaError):
        Inspection.from_dict(inspection_dict(missing_parts=bad))


@pytest.mark.parametrize("path", ["data/a.png", "data\\a.png", "a.png", "../a.png"])
def test_inspection_accepts_relative_image_path(path):
    assert Inspection.from_dict(inspection_dict(image_path=path)).image_path == path


@pytest.mark.parametrize("path", ["/abs/a.png", "C:\\a.png", "C:/a.png", "\\\\server\\share\\a.png", ""])
def test_inspection_rejects_absolute_or_empty_image_path(path):
    with pytest.raises(SchemaError):
        Inspection.from_dict(inspection_dict(image_path=path))


@pytest.mark.parametrize("bad", ["yesterday", "2026-13-40T00:00:00", 12345, None])
def test_inspection_rejects_bad_datetime(bad):
    with pytest.raises(SchemaError):
        Inspection.from_dict(inspection_dict(captured_at=bad))


# ---------- prediction ----------

def test_prediction_valid_and_round_trip():
    d = prediction_dict()
    obj = Prediction.from_dict(d)
    assert obj.predicted_label is Label.NG
    assert obj.to_dict() == d
    assert Prediction.from_dict(obj.to_dict()) == obj


@pytest.mark.parametrize("score", [0.0, 1.0])
def test_prediction_score_bounds_accepted(score):
    label = "NG" if score >= 0.5 else "OK"
    assert Prediction.from_dict(prediction_dict(score=score, predicted_label=label)).score == score


@pytest.mark.parametrize("score", [-0.01, 1.01, float("nan"), "0.5", None, True])
def test_prediction_rejects_bad_score(score):
    with pytest.raises(SchemaError):
        Prediction.from_dict(prediction_dict(score=score))


def test_prediction_label_must_match_score_and_threshold():
    with pytest.raises(SchemaError):
        Prediction.from_dict(prediction_dict(score=0.7, threshold=0.5, predicted_label="OK"))
    with pytest.raises(SchemaError):
        Prediction.from_dict(prediction_dict(score=0.3, threshold=0.5, predicted_label="NG"))


def test_prediction_score_equal_to_threshold_is_ng():
    assert Prediction.from_dict(prediction_dict(score=0.5, threshold=0.5, predicted_label="NG"))
    with pytest.raises(SchemaError):
        Prediction.from_dict(prediction_dict(score=0.5, threshold=0.5, predicted_label="OK"))


def test_prediction_uses_its_own_threshold():
    # 임계값을 0.8 로 올리면 score 0.7 은 OK
    assert Prediction.from_dict(prediction_dict(score=0.7, threshold=0.8, predicted_label="OK"))


@pytest.mark.parametrize("bad", [1.5, True, "12", None])
def test_prediction_rejects_bad_latency(bad):
    with pytest.raises(SchemaError):
        Prediction.from_dict(prediction_dict(latency_ms=bad))


@pytest.mark.parametrize("bad", ["ok", "", None])
def test_prediction_rejects_bad_label(bad):
    with pytest.raises(SchemaError):
        Prediction.from_dict(prediction_dict(predicted_label=bad))


@pytest.mark.parametrize(
    "field",
    ["prediction_id", "inspection_id", "model_version", "predicted_label",
     "score", "threshold", "latency_ms", "predicted_at"],
)
def test_prediction_required_fields(field):
    d = prediction_dict()
    del d[field]
    with pytest.raises(SchemaError):
        Prediction.from_dict(d)


def test_prediction_has_no_stored_correctness_field():
    # 정오 여부는 저장하지 않고 조회 때 계산하는 값 (3장)
    with pytest.raises(SchemaError):
        Prediction.from_dict(prediction_dict(is_correct=True))


# ---------- 추론 출력 ----------

def test_inference_output_valid_and_rejects_bad_values():
    d = {"predicted_label": "OK", "score": 0.1, "model_version": "v1", "latency_ms": 8}
    assert InferenceOutput.from_dict(d).to_dict() == d
    for bad in ({"score": 1.2}, {"predicted_label": "ng"}, {"latency_ms": 1.5}, {"model_version": ""}):
        with pytest.raises(SchemaError):
            InferenceOutput.from_dict({**d, **bad})


# ---------- metrics.json ----------

def test_metrics_valid_and_round_trip():
    d = metrics_dict()
    obj = ModelMetrics.from_dict(d)
    assert isinstance(obj.normal, ClassMetrics)
    assert obj.to_dict() == d
    assert ModelMetrics.from_dict(obj.to_dict()) == obj


@pytest.mark.parametrize("cm", [[[1, 2, 3], [4, 5, 6]], [[1, 2]], [1, 2, 3, 4], [[1.5, 0], [0, 1]], "x"])
def test_metrics_rejects_bad_confusion_matrix(cm):
    d = metrics_dict()
    d["normal"] = class_metrics_dict(confusion_matrix=cm)
    with pytest.raises(SchemaError):
        ModelMetrics.from_dict(d)


@pytest.mark.parametrize("snap", [{"train": -1}, {"train": 1.5}, {"": 1}, [1, 2], None])
def test_metrics_rejects_bad_snapshot(snap):
    with pytest.raises(SchemaError):
        ModelMetrics.from_dict(metrics_dict(train_data_snapshot=snap))


@pytest.mark.parametrize("field", ["version", "threshold", "normal", "drift", "train_data_snapshot", "created_at"])
def test_metrics_required_fields(field):
    d = copy.deepcopy(metrics_dict())
    del d[field]
    with pytest.raises(SchemaError):
        ModelMetrics.from_dict(d)


# ---------- 이미지 규칙 (1장) ----------

def test_image_size_matches_preprocess():
    assert IMAGE_SIZE == preprocess.IMG_SIZE == 128


def test_image_array_valid():
    validate_image_array(np.zeros((128, 128), dtype=np.uint8))


@pytest.mark.parametrize(
    "arr",
    [
        np.zeros((64, 64), dtype=np.uint8),
        np.zeros((128, 128, 3), dtype=np.uint8),
        np.zeros((128, 128, 1), dtype=np.uint8),
        np.zeros((128, 128), dtype=np.float32),
        [[0] * 128] * 128,
    ],
)
def test_image_array_rejects_bad(arr):
    with pytest.raises(SchemaError):
        validate_image_array(arr)


def test_pil_image_valid():
    validate_pil_image(Image.new("L", (128, 128)))


@pytest.mark.parametrize("img", [Image.new("RGB", (128, 128)), Image.new("L", (64, 64)), Image.new("L", (128, 127)), "x"])
def test_pil_image_rejects_bad(img):
    with pytest.raises(SchemaError):
        validate_pil_image(img)
