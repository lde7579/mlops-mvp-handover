"""추론·학습 설정(predictor.toml) 읽기와 검증."""
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("predictor.toml")
_KEYS = {"threshold", "epochs", "batch_size", "learning_rate", "seed", "models_dir"}


class ConfigError(ValueError):
    """설정 파일 값이 규칙에 맞지 않음."""


@dataclass(frozen=True)
class PredictorConfig:
    threshold: float
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    models_dir: str


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _positive_int(name: str, v: Any) -> int:
    if not _is_int(v) or v < 1:
        raise ConfigError(f"{name}: 1 이상의 정수여야 합니다 (받은 값: {v!r})")
    return v


def load_config(path=None) -> PredictorConfig:
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    missing = sorted(_KEYS - raw.keys())
    unknown = sorted(raw.keys() - _KEYS)
    if missing:
        raise ConfigError(f"설정에 없는 항목: {', '.join(missing)}")
    if unknown:
        raise ConfigError(f"알 수 없는 설정 항목: {', '.join(unknown)}")

    threshold = raw["threshold"]
    if not _is_number(threshold) or not 0.0 <= threshold <= 1.0:
        raise ConfigError(f"threshold: 0~1 사이 숫자여야 합니다 (받은 값: {threshold!r})")
    lr = raw["learning_rate"]
    if not _is_number(lr) or not lr > 0:
        raise ConfigError(f"learning_rate: 0보다 큰 숫자여야 합니다 (받은 값: {lr!r})")
    if not _is_int(raw["seed"]):
        raise ConfigError(f"seed: 정수여야 합니다 (받은 값: {raw['seed']!r})")
    models_dir = raw["models_dir"]
    if (not isinstance(models_dir, str) or not models_dir
            or PurePosixPath(models_dir).anchor or PureWindowsPath(models_dir).anchor):
        raise ConfigError(f"models_dir: 프로젝트 루트 기준 상대경로여야 합니다 (받은 값: {models_dir!r})")

    return PredictorConfig(
        threshold=float(threshold),
        epochs=_positive_int("epochs", raw["epochs"]),
        batch_size=_positive_int("batch_size", raw["batch_size"]),
        learning_rate=float(lr),
        seed=raw["seed"],
        models_dir=models_dir,
    )
