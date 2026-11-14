"""모니터링: GET /monitoring/performance, /monitoring/drift 의 계산 (api-v1.md A24·A25).

- 기본 신호는 "정답 라벨 기반 정확도"다. 판정과 정답(검사의 true_label)을 검사 ID 로 짝지어 센다.
  최근 N건과 경고는 기본으로 현재 production 버전의 판정만 본다 (승인·롤백 직후 이전 모델 때문에 경고가 남지 않게).
- data_split 으로 거르지 않는다. 재학습으로 pool 이 train 이 되어도 같은 기간의 수치는 바뀌지 않는다.
- N(30)과 정확도 임계값(0.95)은 5주차 측정으로 확정한 값이다. 응답의 confirmed 는 true 다 (requirements.md D-3, api-v1.md A27).
- 입력 분포(평균 밝기, 노이즈 정도)는 보조 신호이고 값만 돌려준다 (경고 규칙은 5주차).
  기준은 metrics.json 에 없어서(건수·시드뿐) 고정 정상 학습셋(data/dataset/train)에서 균등 간격 300장을 처음 요청 때 한 번 계산해 둔다.
  이미지별 값은 inspection_id 로 기억해 두어서 3~5초마다 불려도 새 이미지만 읽는다.
- 시간 구간 묶음은 파이썬에서 한다 (SQLite 날짜 함수가 마이크로초+오프셋 문자열을 믿기 어렵다). 구간 안의 네 값만 읽는다.
"""
import math
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

from backend.images import resolve_image
from backend.repository import ModelVersionRepository, PredictionRepository, PredictionStats


RECENT_N = 30                  # 확정 (D-3)
ACCURACY_THRESHOLD = 0.95      # 확정 (D-3)
CRITERIA_CONFIRMED = True      # N·T 확정 여부 (응답의 confirmed, api-v1.md A27)
MAX_BUCKETS = 720
DEFAULT_WINDOW = timedelta(hours=1)
BASELINE_DIR = "data/dataset/train"
BASELINE_SAMPLE = 300
CACHE_LIMIT = 2000
SHIFT_NOTE = "shift(위치 이동)는 평균 밝기와 노이즈 정도로는 구별되지 않습니다 (스파이크 결과)"
_VERSION = re.compile(r"^v(\d+)$")


class MonitoringInputError(ValueError):
    """조회 조건이 맞지 않는다 (400). 메시지는 응답에 그대로 쓴다."""


@dataclass(frozen=True)
class Point:
    bucket_start: datetime
    model_version: str
    n: int
    accuracy: float
    recall: Optional[float]


@dataclass(frozen=True)
class PerformanceResult:
    since: datetime
    until: datetime
    bucket_seconds: int
    recent_n: int
    recent_version: Optional[str]
    recent: PredictionStats
    points: list


@dataclass(frozen=True)
class BaselineStats:
    source: str
    n: int
    brightness_mean: float
    brightness_std: float
    noise_mean: float
    noise_std: float


@dataclass(frozen=True)
class DriftResult:
    model_version: Optional[str]
    n: int
    accuracy: Optional[float]
    recent_n: int
    active: Optional[bool]
    window_n: int
    baseline: Optional[BaselineStats]
    current: Optional[tuple]    # (평균 밝기, 평균 노이즈)
    distance: Optional[tuple]   # (밝기 σ거리, 노이즈 σ거리) 각각 계산할 수 없으면 None


def read_pixels(path) -> Optional[np.ndarray]:
    """PNG 를 2차원 실수 배열로 읽는다. 열 수 없거나 흑백 2차원이 아니면 None."""
    try:
        with Image.open(path) as image:
            image.load()
            pixels = np.asarray(image, dtype=np.float64)
    except (OSError, ValueError):
        return None
    return pixels if pixels.ndim == 2 and pixels.shape[1] >= 2 else None


def measure(pixels: np.ndarray) -> tuple:
    """(평균 밝기, 노이즈 정도 = 이웃한 가로 픽셀 값 차이의 평균). 스파이크(results_detect.md 7장)와 같은 정의."""
    return float(pixels.mean()), float(np.abs(np.diff(pixels, axis=1)).mean())


def _recall(tp: int, fn: int) -> Optional[float]:
    return tp / (tp + fn) if tp + fn else None


def _version_key(version: str):
    m = _VERSION.match(version)
    return (0, int(m.group(1)), version) if m else (1, 0, version)


def _sigma(current: Optional[float], mean: float, std: float) -> Optional[float]:
    return None if current is None or std == 0 else (current - mean) / std


class MonitoringService:
    def __init__(self, root, *, clock: Optional[Callable[[], datetime]] = None,
                 reader: Callable[[Path], Optional[np.ndarray]] = read_pixels,
                 baseline_sample: int = BASELINE_SAMPLE, cache_limit: int = CACHE_LIMIT):
        """clock: 현재 시각(UTC)을 돌려주는 함수, reader: 이미지를 읽는 함수. 시험에서 가짜로 바꿔 끼운다."""
        self._root = Path(root)
        self._clock = clock if clock is not None else (lambda: datetime.now(timezone.utc))
        self._reader = reader
        self._baseline_sample = baseline_sample
        self._cache_limit = cache_limit
        self._cache: dict = {}                      # inspection_id -> (밝기, 노이즈)
        self._cache_lock = threading.Lock()
        self._baseline: Optional[BaselineStats] = None
        self._baseline_lock = threading.Lock()

    # ---------- 공통 ----------

    @staticmethod
    def _version(session: Session, model_version: Optional[str]) -> Optional[str]:
        """model_version 을 주면 그 버전, 아니면 현재 production 버전 (없으면 None)."""
        if model_version is not None:
            return model_version
        production = ModelVersionRepository(session).get_production()
        return None if production is None else production.version

    @staticmethod
    def _stats(session: Session, version: Optional[str], recent: int) -> PredictionStats:
        if version is None:
            return PredictionStats(total=0, ng_count=0, tn=0, fp=0, fn=0, tp=0)
        return PredictionRepository(session).stats(recent=recent, model_version=version)

    # ---------- GET /monitoring/performance ----------

    def performance(self, session: Session, *, since: Optional[datetime], until: Optional[datetime], bucket_seconds: int,
                    recent: int, model_version: Optional[str]) -> PerformanceResult:
        end = until if until is not None else self._clock()
        begin = since if since is not None else end - DEFAULT_WINDOW
        begin, end = begin.astimezone(timezone.utc), end.astimezone(timezone.utc)
        if begin >= end:
            raise MonitoringInputError("since 는 until 보다 빨라야 합니다")
        buckets = math.ceil(end.timestamp() / bucket_seconds) - math.floor(begin.timestamp() / bucket_seconds)
        if buckets > MAX_BUCKETS:
            raise MonitoringInputError(f"구간이 너무 많습니다 ({buckets}개, 최대 {MAX_BUCKETS}개). 기간을 줄이거나 bucket_seconds 를 늘려 주세요")

        cells: dict = {}  # (구간 시작 초, 버전) -> [TN, FP, FN, TP]
        for predicted_at, version, predicted, truth in PredictionRepository(session).list_for_window(
                since=begin, until=end, model_version=model_version):
            key = (math.floor(predicted_at.timestamp() / bucket_seconds) * bucket_seconds, version)
            cell = cells.setdefault(key, [0, 0, 0, 0])
            cell[(1 if predicted == "NG" else 0) + (2 if truth == "NG" else 0)] += 1  # OK/OK=0, NG/OK=1, OK/NG=2, NG/NG=3
        points = [Point(datetime.fromtimestamp(start, timezone.utc), version, sum(c), (c[0] + c[3]) / sum(c), _recall(c[3], c[2]))
                  for (start, version), c in sorted(cells.items(), key=lambda kv: (kv[0][0], _version_key(kv[0][1])))]

        version = self._version(session, model_version)
        return PerformanceResult(since=begin, until=end, bucket_seconds=bucket_seconds, recent_n=recent, recent_version=version,
                                 recent=self._stats(session, version, recent), points=points)

    # ---------- GET /monitoring/drift ----------

    def drift(self, session: Session, *, recent: int, model_version: Optional[str]) -> DriftResult:
        version = self._version(session, model_version)
        stats = self._stats(session, version, recent)
        active = None
        if version is not None and stats.n >= recent:  # N건이 모이기 전에는 판단을 미룬다 (초반 헛경고 방지)
            active = stats.accuracy < ACCURACY_THRESHOLD

        values = []
        if version is not None:
            for inspection_id, image_path in PredictionRepository(session).recent_image_paths(recent=recent, model_version=version):
                value = self._inspection_value(inspection_id, image_path)
                if value is not None:
                    values.append(value)
        current = (float(np.mean([v[0] for v in values])), float(np.mean([v[1] for v in values]))) if values else None
        baseline = self._baseline_stats()
        distance = None
        if baseline is not None:
            distance = (_sigma(None if current is None else current[0], baseline.brightness_mean, baseline.brightness_std),
                        _sigma(None if current is None else current[1], baseline.noise_mean, baseline.noise_std))
        return DriftResult(model_version=version, n=stats.n, accuracy=stats.accuracy, recent_n=recent, active=active,
                           window_n=len(values), baseline=baseline, current=current, distance=distance)

    def _inspection_value(self, inspection_id: str, image_path: str) -> Optional[tuple]:
        """검사 이미지 한 장의 (밝기, 노이즈). 한 번 읽은 값은 기억해 둔다. 읽을 수 없으면 None (기억하지 않는다)."""
        with self._cache_lock:
            hit = self._cache.get(inspection_id)
        if hit is not None:
            return hit
        path = resolve_image(self._root, image_path)
        pixels = None if path is None else self._reader(path)
        if pixels is None:
            return None
        value = measure(pixels)
        with self._cache_lock:
            self._cache[inspection_id] = value
            while len(self._cache) > self._cache_limit:
                self._cache.pop(next(iter(self._cache)))  # 가장 오래된 것부터 버린다
        return value

    def _baseline_stats(self) -> Optional[BaselineStats]:
        """고정 정상 학습셋에서 균등 간격으로 고른 이미지의 통계. 처음 한 번만 계산한다 (학습셋이 없으면 None, 다음 요청 때 다시 본다)."""
        with self._baseline_lock:
            if self._baseline is not None:
                return self._baseline
            try:
                names = sorted(n for n in os.listdir(self._root / BASELINE_DIR) if n.lower().endswith(".png"))
            except OSError:
                return None
            if not names:
                return None
            picks = sorted(set(np.linspace(0, len(names) - 1, min(self._baseline_sample, len(names))).astype(int).tolist()))
            values = []
            for i in picks:
                path = resolve_image(self._root, f"{BASELINE_DIR}/{names[i]}")
                pixels = None if path is None else self._reader(path)
                if pixels is not None:
                    values.append(measure(pixels))
            if not values:
                return None
            arr = np.array(values)
            self._baseline = BaselineStats(
                source=BASELINE_DIR, n=len(values), brightness_mean=float(arr[:, 0].mean()), brightness_std=float(arr[:, 0].std()),
                noise_mean=float(arr[:, 1].mean()), noise_std=float(arr[:, 1].std()))
            return self._baseline
