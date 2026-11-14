"""재학습: POST /training/runs (운영자 이상), GET /training/runs, GET /training/runs/{id} (로그인한 누구나). api-v1.md A15, A16."""
import re
from typing import Annotated, Optional

from fastapi import APIRouter, Body, Depends, Query

from backend.api_models import ErrorResponse, TrainingRunList, TrainingRunOut, TrainingRunRequest
from backend.deps import CurrentUser, RetrainDep, SessionDep, User, require_role
from backend.errors import ApiError
from backend.repository import MAX_LIMIT, ModelVersionRepository, RunStatus, RunTrigger, TrainingRunRepository
from backend.retrain import DatasetUnavailableError, NoPoolDataError, RetrainBusyError
from backend.security import Role

router = APIRouter(prefix="/training", tags=["training"])

_ERRORS = {400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}}
_RUN_ID = re.compile(r"RUN-[0-9]+")  # fullmatch 로만 쓴다 (숫자는 ASCII 만, 끝의 줄바꿈도 허용하지 않는다)


def _snapshots(session) -> dict:
    """결과 버전 -> 그 모델의 train_data_snapshot (학습 데이터 요약)."""
    return {m.version: m.train_data_snapshot for m in ModelVersionRepository(session).list()}


@router.post("/runs", status_code=201, response_model=TrainingRunOut,
             responses={**_ERRORS, 403: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
def start_run(session: SessionDep, retrain: RetrainDep, _: Annotated[User, Depends(require_role(Role.OPERATOR))],
              body: Annotated[TrainingRunRequest, Body()] = TrainingRunRequest()) -> TrainingRunOut:
    """재학습을 시작한다. 학습은 응답 뒤에 백그라운드로 이어지며, 응답은 queued 상태의 실행 한 건이다."""
    try:
        run = retrain.start(session, RunTrigger(body.trigger))
    except (RetrainBusyError, NoPoolDataError) as e:
        raise ApiError(409, str(e)) from None
    except DatasetUnavailableError as e:
        raise ApiError(503, str(e)) from None
    return TrainingRunOut.of(run)


@router.get("/runs", response_model=TrainingRunList, responses=_ERRORS)
def list_runs(session: SessionDep, _: CurrentUser,
              status: Annotated[Optional[RunStatus], Query(description="이 상태의 실행만")] = None,
              limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = 50,
              offset: Annotated[int, Query(ge=0)] = 0) -> TrainingRunList:
    """실행 이력, 최신 먼저. total 은 필터를 적용한 전체 건수다."""
    runs = TrainingRunRepository(session)
    snapshots = _snapshots(session)
    items = [TrainingRunOut.of(r, snapshots.get(r.result_version)) for r in runs.list(status=status, limit=limit, offset=offset)]
    return TrainingRunList(items=items, total=runs.count(status=status))


@router.get("/runs/{run_id}", response_model=TrainingRunOut, responses={**_ERRORS, 404: {"model": ErrorResponse}})
def get_run(run_id: str, session: SessionDep, _: CurrentUser) -> TrainingRunOut:
    """실행 한 건. RUN-숫자 형식이 아니거나 없는 ID 는 404 (형식이 틀리면 DB 를 읽지 않는다)."""
    record = TrainingRunRepository(session).get(run_id) if _RUN_ID.fullmatch(run_id) else None
    if record is None:
        raise ApiError(404, "재학습 실행 기록을 찾을 수 없습니다")
    return TrainingRunOut.of(record, _snapshots(session).get(record.result_version))
