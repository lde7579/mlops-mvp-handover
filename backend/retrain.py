"""재학습 실행: POST /training/runs 의 본체. 학습은 요청 뒤에 백그라운드 스레드가 이어서 한다 (api-v1.md A16, 계약 7장 C7·C8).

흐름 (각 단계는 짧은 별도 트랜잭션이라 학습하는 동안 DB 쓰기 잠금을 쥐지 않는다):
  요청 안: 사전 확인 -> training_run(queued) 만들고 커밋 -> 스레드 시작
  data     running + started_at. DB 의 train·pool 검사와 고정 데이터셋(train, test_normal, test_drift)을 읽는다.
           편입 대상은 여기서 읽은 pool ID 집합으로 고정한다 (학습 중에 새로 생긴 pool 행은 pool 로 남는다).
  train    parts.predictor.train() 호출. 이 함수가 학습과 평가를 한 번에 하고 models/<version>/ 에 저장한다.
  evaluate train() 이 돌려준 결과를 확인하는 짧은 단계다 (실제 평가는 train() 안에서 끝났다).
  gate     한 트랜잭션: 후보 등록(model_version) + 게이트 평가·승인 요청(submit_candidate) + pool 편입 + succeeded.
           게이트 미통과도 succeeded 다. 하나라도 실패하면 모두 되돌리고 failed 로 기록한다 (편입도 일어나지 않는다).
- failed 는 학습 도중 오류일 때만이다 (계약 7장). 만들다 만 models/<version>/ 폴더는 지우지 않는다 (다음 번호 계산에는 포함).
- 프로세스 하나(워커 1개)만 지원한다. 서버가 학습 중에 꺼지면 queued·running 으로 남은 기록을 다음 시작 때 failed 로 정리한다.
- 추론(InferenceService)과 락을 공유하지 않는다. 승인·핫리로드는 4-C 몫이다.
"""
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from backend.database import session_scope
from backend.images import resolve_image
from backend.promotion import submit_candidate
from backend.registry import VERSION_NAME, models_root
from backend.repository import (
    ConflictError,
    InspectionRepository,
    ModelVersionRepository,
    RunStatus,
    RunStep,
    RunTrigger,
    TrainingRunRecord,
    TrainingRunRepository,
)
from backend.settings import Settings
from services.training.dataset import TrainingDataError, dataset_available, load_fixed_sets, read_png
from services.training.gate import evaluate_gate

log = logging.getLogger("backend")

MIN_POOL = 1  # 재학습에 필요한 신규 데이터(pool) 최소 건수
BUSY = "이미 실행 중인 재학습이 있습니다"
NO_POOL = "재학습에 쓸 신규 데이터가 없습니다"
NO_DATASET = "고정 데이터셋이 없습니다"
INTERRUPTED = "서버가 재시작되어 중단되었습니다"
MAX_MESSAGE = 500


class RetrainBusyError(ConflictError):
    """queued 또는 running 인 재학습이 이미 있다 (409)."""


class NoPoolDataError(ConflictError):
    """재학습에 쓸 신규 데이터(pool)가 없다 (409)."""


class DatasetUnavailableError(RuntimeError):
    """고정 데이터셋이 없어 재학습을 시작할 수 없다 (503)."""


@dataclass
class TrainingInputs:
    images: object        # uint8 (N, 128, 128): 고정 train + DB train + DB pool 순서
    labels: list
    eval_normal: tuple    # (이미지 배열, 라벨 목록)
    eval_drift: tuple
    snapshot: dict        # {"train": 건수, "pool": 건수}  (합이 이미지 수)
    pool_ids: list        # data 단계에서 읽은 pool 검사 ID (이 목록만 train 으로 편입한다)


def _default_trainer(*args, **kwargs):
    from parts.predictor import train  # 여기서 처음 torch 를 가져온다

    return train(*args, **kwargs)


class RetrainService:
    def __init__(self, settings: Settings, session_factory, *, trainer: Optional[Callable] = None,
                 fixed_loader: Callable = load_fixed_sets, clock: Optional[Callable[[], datetime]] = None):
        self._root = Path(settings.root)
        self._factory = session_factory
        self._trainer = trainer if trainer is not None else _default_trainer
        self._fixed_loader = fixed_loader
        self._clock = clock if clock is not None else (lambda: datetime.now().astimezone())
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    # ---------- 요청 안 ----------

    def start(self, session, trigger: RunTrigger) -> TrainingRunRecord:
        """사전 확인 후 queued 기록을 만들어 커밋하고 백그라운드 학습을 시작한다.

        RetrainBusyError(이미 실행 중), DatasetUnavailableError(데이터셋 없음), NoPoolDataError(pool 0건) 이면 아무것도 만들지 않는다.
        """
        with self._lock:
            runs = TrainingRunRepository(session)
            if runs.active() is not None or (self._thread is not None and self._thread.is_alive()):
                raise RetrainBusyError(BUSY)
            if not dataset_available(self._root):
                raise DatasetUnavailableError(NO_DATASET)
            if InspectionRepository(session).count(data_split="pool") < MIN_POOL:
                raise NoPoolDataError(NO_POOL)
            run = runs.add(trigger)
            session.commit()  # 스레드가 이 기록을 볼 수 있어야 한다
            self._thread = threading.Thread(target=self._run, args=(run.run_id,), name=f"retrain-{run.run_id}", daemon=True)
            self._thread.start()
            return run

    def wait(self, timeout: Optional[float] = None) -> bool:
        """진행 중인 학습이 끝날 때까지 기다린다 (시험과 수동 확인용). 끝났으면(또는 없으면) True."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            return not thread.is_alive()
        return True

    def mark_interrupted(self) -> int:
        """서버가 켜질 때: queued·running 으로 남은 기록을 failed 로 정리한다 (api-v1.md 4장). 지운 모델 폴더는 없다."""
        with session_scope(self._factory) as session:
            return TrainingRunRepository(session).fail_unfinished(INTERRUPTED, self._clock())

    # ---------- 백그라운드 ----------

    def _set(self, run_id: str, **fields) -> None:
        with session_scope(self._factory) as session:
            TrainingRunRepository(session).update(run_id, **fields)

    def _run(self, run_id: str) -> None:
        try:
            self._set(run_id, status=RunStatus.RUNNING, started_at=self._clock(), step=RunStep.DATA)
            with session_scope(self._factory) as session:
                inputs = self._collect(session)
                version = self._next_version(session)
            self._set(run_id, step=RunStep.TRAIN)
            metrics = self._trainer(
                inputs.images, inputs.labels, version=version, eval_normal=inputs.eval_normal, eval_drift=inputs.eval_drift,
                snapshot=inputs.snapshot, root=self._root)
            self._set(run_id, step=RunStep.EVALUATE)
            if metrics.version != version:
                raise RuntimeError(f"학습 결과의 버전({metrics.version})이 요청한 버전({version})과 다릅니다")
            self._set(run_id, step=RunStep.GATE)
            self._finish(run_id, metrics, inputs)
        except Exception as e:
            log.exception("재학습 %s 이(가) 실패했습니다", run_id)
            self._fail(run_id, e)

    def _collect(self, session) -> TrainingInputs:
        """학습에 쓸 데이터를 읽는다. 읽기만 한다 (편입은 학습이 성공한 뒤 gate 단계에서)."""
        inspections = InspectionRepository(session).list_for_training()  # data_split 이 train 또는 pool 인 것만 (테스트셋은 포함되지 않는다)
        db_train = [i for i in inspections if i.data_split.value == "train"]
        pool = [i for i in inspections if i.data_split.value == "pool"]
        if len(pool) < MIN_POOL:
            raise TrainingDataError(NO_POOL)
        fixed = self._fixed_loader(self._root)
        arrays, labels = [fixed["train"].images], list(fixed["train"].labels)
        for i in db_train + pool:  # 고정 train, DB train, DB pool 순서
            path = resolve_image(self._root, i.image_path)
            if path is None:
                raise TrainingDataError(f"{i.inspection_id}: 이미지를 찾을 수 없습니다 ({i.image_path})")
            arrays.append(read_png(path)[np.newaxis])
            labels.append(i.true_label.value)
        return TrainingInputs(
            images=np.concatenate(arrays), labels=labels,
            eval_normal=(fixed["test_normal"].images, list(fixed["test_normal"].labels)),
            eval_drift=(fixed["test_drift"].images, list(fixed["test_drift"].labels)),
            snapshot={"train": len(fixed["train"].labels) + len(db_train), "pool": len(pool)},
            pool_ids=[i.inspection_id for i in pool])

    def _next_version(self, session) -> str:
        """DB 의 model_version 과 models/ 폴더의 v숫자 중 가장 큰 번호 + 1 (등록 안 된 폴더와 부딪히지 않게 폴더도 본다)."""
        numbers = [int(r.version[1:]) for r in ModelVersionRepository(session).list() if VERSION_NAME.match(r.version)]
        folder = models_root(self._root)
        if folder.is_dir():
            numbers += [int(p.name[1:]) for p in folder.iterdir() if p.is_dir() and VERSION_NAME.match(p.name)]
        return f"v{max(numbers, default=0) + 1}"

    def _finish(self, run_id: str, metrics, inputs: TrainingInputs) -> None:
        """후보 등록 + 게이트 + pool 편입 + succeeded 를 한 트랜잭션으로 한다. 하나라도 실패하면 모두 일어나지 않는다."""
        now = self._clock()
        with session_scope(self._factory) as session:
            models = ModelVersionRepository(session)
            record = models.register(metrics)  # candidate
            baseline = models.get_production()
            gate = evaluate_gate(None if baseline is None else baseline.metrics, record.metrics)
            submit_candidate(session, record.version, gate, now=now, training_run_id=run_id)
            InspectionRepository(session).merge_pool_into_train(inputs.pool_ids)
            TrainingRunRepository(session).update(
                run_id, status=RunStatus.SUCCEEDED, step=RunStep.GATE, result_version=record.version, finished_at=now)

    def _fail(self, run_id: str, error: Exception) -> None:
        message = (str(error).strip() or type(error).__name__)[:MAX_MESSAGE]
        try:
            self._set(run_id, status=RunStatus.FAILED, error_message=message, finished_at=self._clock())
        except Exception:
            log.exception("재학습 %s 의 실패를 기록하지 못했습니다", run_id)
