"""검사 1건 생성 (POST /simulator/generate의 본체): 수집기로 이미지를 만들고 DB 에 등록한다.

- 요청 하나씩만 처리한다 (락). 번호(ID) 발급과 파일 저장이 겹치지 않게 하기 위해서다. **프로세스 하나(워커 1개)만 지원**한다.
- 수집기(Collector)는 첫 요청 때 만든다 (앱을 만들 때는 파일도 폴더도 만들지 않는다).
- 번호: 수집기는 data/<분류>/ 폴더 안의 파일만 보고 번호를 정한다. 그래서 매번 DB 의 최대 번호를 확인해 그보다 크게 맞춘다
  (max(파일, DB) + 1). 수집기의 내부 속성 _next_id 를 직접 올리는 임시 방법이다 (tests/backend/test_simulator_generate.py 의 마지막 시험들이
  이 속성이 바뀌면 바로 깨지도록 지킨다).
- 파일을 먼저 저장하고 DB 에 등록한다. DB 등록이나 커밋이 실패하면 방금 만든 PNG 를 지워서 고아 파일을 남기지 않는다.
  커밋을 이 안에서 하는 이유: 요청이 끝난 뒤 커밋하면 커밋 실패 때 파일을 지울 방법이 없다.
"""
import logging
import threading
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from backend.repository import DuplicateError, InspectionRepository
from backend.settings import Settings
from common.schemas import Inspection
from parts.collector.collector import Collector

log = logging.getLogger("backend")

ID_ATTEMPTS = 3  # 번호가 겹칠 때 다시 시도하는 횟수


class GenerationService:
    def __init__(self, settings: Settings):
        self._root = Path(settings.root)
        self._config_path = Path(settings.collector_config)
        self._lock = threading.Lock()
        self._collector: Optional[Collector] = None

    def _ready(self, db_max: int) -> Collector:
        """수집기를 (처음이면 만들고) 다음 번호를 max(파일, DB) + 1 로 맞춘다."""
        if self._collector is None:
            probe = Collector(self._config_path, root=self._root)  # 폴더 안 파일 기준 시작 번호를 알아내는 용도
            start = max(probe._next_id, db_max + 1)
            # 시드 = 설정 시드 + 시작 번호: 같은 번호에서 시작하면 같은 이미지가 나온다 (재현 가능)
            self._collector = Collector(self._config_path, root=self._root, seed=probe.config.seed + start)
        self._collector._next_id = max(self._collector._next_id, db_max + 1)
        return self._collector

    def _remove_png(self, inspection: Inspection) -> None:
        try:
            (self._root / inspection.image_path).unlink(missing_ok=True)
        except OSError as e:
            log.warning("생성 실패 뒤 PNG 를 지우지 못했습니다 (%s): %s", inspection.image_path, e)

    def generate(self, session: Session, scenario: str, *, label: Optional[str] = None,
                 drift_type: Optional[str] = None) -> Inspection:
        """검사 1건을 만들어 저장하고 커밋한다 (data_split 은 항상 pool).

        label(OK/NG), drift_type(dark/noise/shift)을 주면 수집기의 공개 옵션으로 그대로 넘긴다 (자동 생성이 불량률과 드리프트 종류를 정할 때 쓴다).
        """
        with self._lock:
            repo = InspectionRepository(session)
            for _ in range(ID_ATTEMPTS):
                collector = self._ready(repo.max_id())
                sample = collector.next_sample(scenario=scenario, data_split="pool", label=label, drift_type=drift_type)
                try:
                    repo.add(sample.inspection)
                    session.commit()
                except DuplicateError:
                    session.rollback()
                    self._remove_png(sample.inspection)
                    continue  # 다음 시도에서 DB 최대 번호를 다시 읽는다
                except BaseException:
                    session.rollback()
                    self._remove_png(sample.inspection)
                    raise
                return sample.inspection
            raise RuntimeError(f"검사 번호를 {ID_ATTEMPTS}번 시도했지만 계속 겹쳤습니다")
