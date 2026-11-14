"""수집기 설정(collector.toml) 읽기와 검증."""
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from common.schemas import IMAGE_SIZE
from parts.collector.simulator import BOLT_COUNT, DRIFT_TYPES

DEFAULT_CONFIG_PATH = Path(__file__).with_name("collector.toml")
_KEYS = {"defect_rate", "image_size", "part_count", "seed", "data_dir", "drift_strength"}


class ConfigError(ValueError):
    """설정 파일 값이 규칙에 맞지 않음."""


@dataclass(frozen=True)
class CollectorConfig:
    defect_rate: float
    image_size: int
    part_count: int
    seed: int
    data_dir: str
    drift_strength: dict


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _ratio(name: str, v: Any) -> float:
    if not _is_number(v) or not 0.0 <= v <= 1.0:
        raise ConfigError(f"{name}: 0~1 사이 숫자여야 합니다 (받은 값: {v!r})")
    return float(v)


def load_config(path=None) -> CollectorConfig:
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    missing = sorted(_KEYS - raw.keys())
    unknown = sorted(raw.keys() - _KEYS)
    if missing:
        raise ConfigError(f"설정에 없는 항목: {', '.join(missing)}")
    if unknown:
        raise ConfigError(f"알 수 없는 설정 항목: {', '.join(unknown)}")

    if raw["image_size"] != IMAGE_SIZE or not _is_int(raw["image_size"]):
        raise ConfigError(f"image_size: 계약 고정값 {IMAGE_SIZE} 이어야 합니다 (받은 값: {raw['image_size']!r})")
    if raw["part_count"] != BOLT_COUNT or not _is_int(raw["part_count"]):
        raise ConfigError(f"part_count: 볼트는 {BOLT_COUNT}개로 고정입니다 (받은 값: {raw['part_count']!r})")
    if not _is_int(raw["seed"]):
        raise ConfigError(f"seed: 정수여야 합니다 (받은 값: {raw['seed']!r})")

    data_dir = raw["data_dir"]
    if not isinstance(data_dir, str) or not data_dir or PurePosixPath(data_dir).anchor or PureWindowsPath(data_dir).anchor:
        raise ConfigError(f"data_dir: 프로젝트 루트 기준 상대경로여야 합니다 (받은 값: {data_dir!r})")

    strength = raw["drift_strength"]
    if not isinstance(strength, dict) or set(strength) != set(DRIFT_TYPES):
        raise ConfigError(f"drift_strength: {', '.join(DRIFT_TYPES)} 3개가 모두 있어야 합니다 (받은 값: {strength!r})")

    return CollectorConfig(
        defect_rate=_ratio("defect_rate", raw["defect_rate"]),
        image_size=raw["image_size"],
        part_count=raw["part_count"],
        seed=raw["seed"],
        data_dir=data_dir,
        drift_strength={k: _ratio(f"drift_strength.{k}", v) for k, v in strength.items()},
    )
