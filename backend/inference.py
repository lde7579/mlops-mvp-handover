"""추론 서비스: 운영(production) 모델을 불러와 두고 이미지 1장을 판정한다 (parts/predictor 의 공개 인터페이스만 사용).

- 앱을 만들 때(create_app)는 아무것도 불러오지 않는다. 서버가 켜질 때(lifespan) 미리 불러오고(preload, 실패해도 서버는 뜬다),
  /predict 때 아직 안 불러왔거나 production 이 바뀌었으면 그때 불러온다.
- torch 는 모델을 처음 불러올 때만 가져온다 (/health 나 검사 조회만 쓰는 동안은 가져오지 않는다).
- 불러오기/교체(reload)는 락으로 하나씩 한다. 판정(predict)은 락 없이 동시에 해도 된다
  (Predictor 가 모델·버전·임계값을 한 번에 통째로 바꾼다).
- 불러오기에 실패하면 503 이고 요청마다 다시 시도한다 (파일을 고치면 서버를 다시 켜지 않아도 복구된다).
  같은 실패는 로그를 한 번만 남긴다.
- 4주차 승인(핫리로드)은 "먼저 읽어 두기 -> 커밋 -> 교체" 순서다 (api-v1.md 4장, A18):
  prepare(version) 이 새 모델을 별도 Predictor 에 읽어 두고(실패하면 ModelUnavailableError, 기존 모델은 그대로),
  DB 커밋이 성공한 뒤 activate(prepared) 가 객체만 바꾼다(메모리에서 포인터 교체라 실패할 일이 없다).
  커밋은 됐는데 activate 가 실행되지 못해도, 다음 ensure_loaded 가 읽어 둔 모델(없으면 파일)을 DB 의 production 기준으로 쓴다.
  reload(version) 은 이미 올라와 있어도 파일에서 다시 불러온다 (운영자용 수동 복구).
"""
import logging
import threading
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from backend.database import session_scope
from backend.repository import ModelVersionRepository
from backend.settings import Settings
from common.schemas import InferenceOutput

log = logging.getLogger("backend")

NO_MODEL = "운영 중인 모델이 없습니다"
LOAD_FAILED = "운영 모델을 불러오지 못했습니다"


class ModelUnavailableError(Exception):
    """지금은 판정할 수 없다 (운영 모델 없음, 불러오기 실패). 메시지는 응답에 그대로 쓴다."""


def _parts():
    from parts.predictor import ModelLoadError, Predictor  # 여기서 처음 torch 를 가져온다

    return Predictor, ModelLoadError


class InferenceService:
    def __init__(self, settings: Settings, *, parts_loader=None):
        """parts_loader: (Predictor 클래스, ModelLoadError 클래스)를 돌려주는 함수. 시험에서 가짜로 바꿔 끼운다 (기본은 parts.predictor)."""
        self._root = Path(settings.root)
        self._parts = parts_loader if parts_loader is not None else _parts
        self._lock = threading.Lock()
        self._predictor = None
        self._prepared = None                        # prepare 로 읽어 둔, 아직 교체하지 않은 Predictor (한 개)
        self._last_failure: Optional[tuple] = None  # 같은 실패를 로그에 반복하지 않기 위한 기억
        self._thresholds: dict[str, float] = {}      # 불러온 버전 -> 그 모델에 기록된 임계값 (불러올 때 락 안에서 기록)

    @property
    def loaded_version(self) -> Optional[str]:
        """지금 불러와 둔 모델 버전 (없으면 None). 파일을 읽거나 torch 를 가져오지 않는다."""
        predictor = self._predictor
        return predictor.version if predictor is not None else None

    def _load(self, version: str, *, force: bool) -> None:
        Predictor, ModelLoadError = self._parts()
        with self._lock:
            if self._predictor is not None and not force and self._predictor.version == version:
                return  # 기다리는 사이에 다른 요청이 이미 불러왔다
            if not force and self._prepared is not None and self._prepared.version == version:
                self._swap(self._prepared)  # 승인이 읽어 둔 모델이 있으면 파일을 다시 읽지 않고 그것을 쓴다
                return
            if self._predictor is None:
                self._predictor = Predictor(root=self._root)
            try:
                self._predictor.load(version)
            except (ModelLoadError, ValueError) as e:  # ValueError: 버전 이름 형식이 올바르지 않음
                self._record_failure(version, e)
                raise ModelUnavailableError(LOAD_FAILED) from e
            self._thresholds[self._predictor.version] = self._predictor.threshold  # 락 안이라 방금 불러온 값과 짝이 맞다
            self._last_failure = None

    def _record_failure(self, version: str, error: Exception) -> None:
        key = (version, str(error))
        if key != self._last_failure:  # 같은 실패는 한 번만 경고한다 (요청마다 다시 시도하지만 로그는 늘리지 않는다)
            log.warning("운영 모델 %s 을 불러오지 못했습니다: %s", version, error)
        self._last_failure = key

    def _swap(self, predictor) -> None:
        """락을 쥔 채 부른다. 판정에 쓸 모델을 통째로 바꾸고 임계값 표에 짝을 기록한다."""
        self._thresholds[predictor.version] = predictor.threshold
        self._predictor = predictor
        if self._prepared is predictor:
            self._prepared = None
        self._last_failure = None

    def prepare(self, version: str):
        """새 모델을 별도 Predictor 에 읽어 둔다 (지금 쓰는 모델과 DB 는 건드리지 않는다). 실패하면 ModelUnavailableError.

        돌려준 객체를 커밋 뒤에 activate 로 넘기고, 커밋하지 못했으면 discard 로 버린다. 락을 쥐지 않고 읽으므로 판정을 막지 않는다.
        """
        Predictor, ModelLoadError = self._parts()
        candidate = Predictor(root=self._root)
        try:
            candidate.load(version)
        except (ModelLoadError, ValueError) as e:  # ValueError: 버전 이름 형식이 올바르지 않음
            log.warning("새 모델 %s 을 읽어 두지 못했습니다: %s", version, e)
            raise ModelUnavailableError(LOAD_FAILED) from e
        with self._lock:
            self._prepared = candidate
        return candidate

    def activate(self, prepared) -> None:
        """커밋이 끝난 뒤 읽어 둔 모델로 바꾼다. 판정 중인 요청은 끝까지 이전 모델을 쓰고(임계값도 이전 것), 이후 요청부터 새 모델을 쓴다."""
        with self._lock:
            self._swap(prepared)

    def discard(self, prepared) -> None:
        """커밋하지 못해 쓰지 않게 된, 읽어 둔 모델을 버린다."""
        with self._lock:
            if self._prepared is prepared:
                self._prepared = None

    def reload(self, version: str) -> None:
        """version 을 (이미 올라와 있어도) 다시 불러온다. 실패하면 ModelUnavailableError 이고 기존 모델은 그대로다."""
        self._load(version, force=True)

    def ensure_loaded(self, session: Session):
        """DB 의 production 버전이 불러와져 있게 하고 Predictor 를 돌려준다."""
        production = ModelVersionRepository(session).get_production()
        if production is None:
            raise ModelUnavailableError(NO_MODEL)
        predictor = self._predictor
        if predictor is None or predictor.version != production.version:
            self._load(production.version, force=False)
        return self._predictor

    def predict(self, session: Session, image) -> tuple[InferenceOutput, float]:
        """(판정 결과, 그 판정에 쓴 임계값). 임계값은 불러온 모델의 값이다 (계약 C5)."""
        predictor = self.ensure_loaded(session)
        output = predictor.predict(image)
        # 판정하는 동안 모델이 교체되어도 짝이 맞도록, 판정에 쓴 버전(output.model_version)의 임계값을 기록해 둔 표에서 찾는다
        return output, self._thresholds[output.model_version]

    def preload(self, session_factory) -> None:
        """서버 시작 때 운영 모델을 미리 불러온다. 어떤 이유로든 실패해도 서버 시작을 막지 않는다."""
        try:
            with session_scope(session_factory) as session:
                self.ensure_loaded(session)
        except ModelUnavailableError as e:
            log.warning("시작할 때 운영 모델을 불러오지 못했습니다: %s", e)
        except Exception:
            log.exception("시작할 때 운영 모델을 확인하는 중 오류가 났습니다")
