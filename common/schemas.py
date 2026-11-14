"""계약 스키마 — docs/data-contract.md v1.0 의 inspection, prediction, 추론 출력, metrics.json 을 코드로 옮긴 것.

문서에 없는 필드·값·범위는 만들지 않는다. 문서가 바뀌면 이 파일도 문서를 따라 바꾼다.
저장소·파일이 있어야 하는 검사(ID 중복, 파일이 열리는가)는 여기서 하지 않는다.
"""
import dataclasses
import numbers
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Optional

import numpy as np
from PIL import Image

# preprocess.IMG_SIZE 와 같은 값이어야 한다 (tests/test_schemas.py 가 확인).
# preprocess 를 import 하면 torch 까지 불러오므로 여기서는 따로 둔다.
IMAGE_SIZE = 128


class SchemaError(ValueError):
    """계약 스키마 검증 실패."""


class Label(StrEnum):
    OK = "OK"
    NG = "NG"


class DataSplit(StrEnum):
    TRAIN = "train"
    POOL = "pool"
    TEST_NORMAL = "test_normal"
    TEST_DRIFT = "test_drift"


class Scenario(StrEnum):
    NORMAL = "normal"
    DRIFT = "drift"


# ---------- 검사 도우미 ----------

def _nonempty_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{name}: 비어 있지 않은 문자열이어야 합니다 (받은 값: {value!r})")
    return value


def _enum(name: str, enum_cls: type[Enum], value: Any):
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError:
            pass
    allowed = ", ".join(m.value for m in enum_cls)
    raise SchemaError(f"{name}: 허용값은 {allowed} 입니다 (받은 값: {value!r})")


def _datetime(name: str, value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    raise SchemaError(f"{name}: ISO 8601 시각이어야 합니다 (받은 값: {value!r})")


def _number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise SchemaError(f"{name}: 숫자여야 합니다 (받은 값: {value!r})")
    return float(value)


def _int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise SchemaError(f"{name}: 정수여야 합니다 (받은 값: {value!r})")
    return int(value)


def _score(name: str, value: Any) -> float:
    score = _number(name, value)
    if not 0.0 <= score <= 1.0:  # NaN 도 여기서 걸린다
        raise SchemaError(f"{name}: 0.0~1.0 이어야 합니다 (받은 값: {value!r})")
    return score


def _relative_path(name: str, value: Any) -> str:
    path = _nonempty_str(name, value)
    # 문서: 프로젝트 루트 기준 상대경로. 윈도우(C:\, \\서버)·유닉스(/) 절대경로를 모두 거부한다.
    if PurePosixPath(path).anchor or PureWindowsPath(path).anchor:
        raise SchemaError(f"{name}: 프로젝트 루트 기준 상대경로여야 합니다 (받은 값: {value!r})")
    return path


def _int_list(name: str, value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise SchemaError(f"{name}: 정수 목록이어야 합니다 (받은 값: {value!r})")
    return [_int(f"{name}[{i}]", v) for i, v in enumerate(value)]


def _build(cls, data: Any):
    """dict 에서 만든다. 필수 필드가 빠지거나 문서에 없는 필드가 있으면 거부한다."""
    if not isinstance(data, Mapping):
        raise SchemaError(f"{cls.__name__}: dict 여야 합니다 (받은 값: {data!r})")
    fields = dataclasses.fields(cls)
    names = {f.name for f in fields}
    required = {f.name for f in fields if f.default is dataclasses.MISSING}
    missing = sorted(required - data.keys())
    unknown = sorted(data.keys() - names)
    if missing:
        raise SchemaError(f"{cls.__name__}: 필수 필드가 없습니다: {', '.join(missing)}")
    if unknown:
        raise SchemaError(f"{cls.__name__}: 문서에 없는 필드입니다: {', '.join(unknown)}")
    return cls(**data)


def _plain(value: Any) -> Any:
    """JSON 으로 저장할 수 있는 값으로 바꾼다 (Enum -> 값, datetime -> ISO 8601)."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


class _Schema:
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        return _build(cls, data)

    def to_dict(self) -> dict[str, Any]:
        return _plain(dataclasses.asdict(self))


# ---------- 2장 inspection ----------

@dataclass
class Inspection(_Schema):
    inspection_id: str
    captured_at: datetime
    image_path: str
    true_label: Label
    missing_parts: list[int]
    scenario: Scenario
    data_split: DataSplit
    drift_type: Optional[str] = None  # "dark / noise / shift 등" 이라 값은 제한하지 않는다

    def __post_init__(self):
        self.inspection_id = _nonempty_str("inspection_id", self.inspection_id)
        self.captured_at = _datetime("captured_at", self.captured_at)
        self.image_path = _relative_path("image_path", self.image_path)
        self.true_label = _enum("true_label", Label, self.true_label)
        self.missing_parts = _int_list("missing_parts", self.missing_parts)
        self.scenario = _enum("scenario", Scenario, self.scenario)
        self.data_split = _enum("data_split", DataSplit, self.data_split)
        if self.drift_type is not None:
            self.drift_type = _nonempty_str("drift_type", self.drift_type)
        if self.true_label is Label.OK and self.missing_parts:
            raise SchemaError("missing_parts: true_label 이 OK 이면 비어 있어야 합니다")


# ---------- 3장 prediction ----------

@dataclass
class Prediction(_Schema):
    prediction_id: str
    inspection_id: str
    model_version: str
    predicted_label: Label
    score: float
    threshold: float
    latency_ms: int
    predicted_at: datetime

    def __post_init__(self):
        self.prediction_id = _nonempty_str("prediction_id", self.prediction_id)
        self.inspection_id = _nonempty_str("inspection_id", self.inspection_id)
        self.model_version = _nonempty_str("model_version", self.model_version)
        self.predicted_label = _enum("predicted_label", Label, self.predicted_label)
        self.score = _score("score", self.score)
        self.threshold = _number("threshold", self.threshold)
        self.latency_ms = _int("latency_ms", self.latency_ms)
        self.predicted_at = _datetime("predicted_at", self.predicted_at)
        # 1장: score 가 임계값 이상이면 NG
        expected = Label.NG if self.score >= self.threshold else Label.OK
        if self.predicted_label is not expected:
            raise SchemaError(
                f"predicted_label: score {self.score} 와 threshold {self.threshold} 로는 "
                f"{expected.value} 여야 합니다 (받은 값: {self.predicted_label.value})"
            )


# ---------- 4장 모델 추론 출력 ----------

@dataclass
class InferenceOutput(_Schema):
    predicted_label: Label
    score: float
    model_version: str
    latency_ms: int

    def __post_init__(self):
        self.predicted_label = _enum("predicted_label", Label, self.predicted_label)
        self.score = _score("score", self.score)
        self.model_version = _nonempty_str("model_version", self.model_version)
        self.latency_ms = _int("latency_ms", self.latency_ms)


# ---------- 4장 metrics.json ----------

@dataclass
class ClassMetrics(_Schema):
    """test_normal 또는 test_drift 한 세트의 결과."""
    accuracy: float
    recall: float
    precision: float
    f1: float
    confusion_matrix: list[list[int]]  # [[TN, FP], [FN, TP]]
    n: int

    def __post_init__(self):
        for name in ("accuracy", "recall", "precision", "f1"):
            setattr(self, name, _number(name, getattr(self, name)))
        cm = self.confusion_matrix
        if not (isinstance(cm, (list, tuple)) and len(cm) == 2
                and all(isinstance(row, (list, tuple)) and len(row) == 2 for row in cm)):
            raise SchemaError(f"confusion_matrix: [[TN, FP], [FN, TP]] 형태의 2×2 여야 합니다 (받은 값: {cm!r})")
        self.confusion_matrix = [[_int("confusion_matrix", v) for v in row] for row in cm]
        self.n = _int("n", self.n)


@dataclass
class ModelMetrics(_Schema):
    version: str
    threshold: float
    normal: ClassMetrics
    drift: ClassMetrics
    train_data_snapshot: dict[str, int]  # 키 이름은 문서에 정해져 있지 않아 제한하지 않는다
    created_at: datetime

    def __post_init__(self):
        self.version = _nonempty_str("version", self.version)
        self.threshold = _number("threshold", self.threshold)
        for name in ("normal", "drift"):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                value = ClassMetrics.from_dict(value)
            elif not isinstance(value, ClassMetrics):
                raise SchemaError(f"{name}: ClassMetrics 또는 dict 여야 합니다 (받은 값: {value!r})")
            setattr(self, name, value)
        snap = self.train_data_snapshot
        if not isinstance(snap, Mapping):
            raise SchemaError(f"train_data_snapshot: dict 여야 합니다 (받은 값: {snap!r})")
        clean = {}
        for key, count in snap.items():
            _nonempty_str("train_data_snapshot 키", key)
            clean[key] = _int(f"train_data_snapshot[{key}]", count)
            if clean[key] < 0:
                raise SchemaError(f"train_data_snapshot[{key}]: 0 이상이어야 합니다 (받은 값: {count!r})")
        self.train_data_snapshot = clean
        self.created_at = _datetime("created_at", self.created_at)


# ---------- 1장 이미지 규칙 (128×128, 흑백 1채널, uint8) ----------

def validate_image_array(arr: Any) -> None:
    """numpy 이미지: shape (128, 128), dtype uint8."""
    if not isinstance(arr, np.ndarray):
        raise SchemaError(f"이미지: numpy 배열이어야 합니다 (받은 값: {type(arr).__name__})")
    if arr.dtype != np.uint8:
        raise SchemaError(f"이미지: dtype 은 uint8 이어야 합니다 (받은 값: {arr.dtype})")
    if arr.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise SchemaError(f"이미지: shape 은 ({IMAGE_SIZE}, {IMAGE_SIZE}) 여야 합니다 (받은 값: {arr.shape})")


def validate_pil_image(img: Any) -> None:
    """PIL 이미지: 모드 L(흑백 1채널), 크기 128×128."""
    if not isinstance(img, Image.Image):
        raise SchemaError(f"이미지: PIL 이미지여야 합니다 (받은 값: {type(img).__name__})")
    if img.mode != "L":
        raise SchemaError(f"이미지: 모드는 L(흑백 1채널)이어야 합니다 (받은 값: {img.mode})")
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        raise SchemaError(f"이미지: 크기는 {IMAGE_SIZE}×{IMAGE_SIZE} 여야 합니다 (받은 값: {img.size})")
