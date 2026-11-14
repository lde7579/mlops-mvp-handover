"""판정 조회: GET /predictions (이력), GET /predictions/stats (통계). 로그인한 누구나."""
from typing import Annotated, Literal, Optional

from fastapi import APIRouter, Query

from backend.api_models import ErrorResponse, PredictionList, PredictionStatsOut, PredictionWithTruthOut
from backend.deps import CurrentUser, SessionDep
from backend.repository import DEFAULT_RECENT, MAX_LIMIT, PredictionRepository

router = APIRouter(prefix="/predictions", tags=["predictions"])

_ERRORS = {400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}}


@router.get("", response_model=PredictionList, responses=_ERRORS)
def list_predictions(
    session: SessionDep, _: CurrentUser,
    model_version: Annotated[Optional[str], Query(min_length=1, max_length=64, description="이 모델 버전의 판정만")] = None,
    inspection_id: Annotated[Optional[str], Query(min_length=1, max_length=64, description="이 검사의 판정만")] = None,
    wrong: Annotated[Optional[Literal["true", "false"]], Query(description="true 는 오판만, false 는 정판만 (소문자만 허용)")] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> PredictionList:
    """판정 이력, 최신 먼저. 항목마다 검사의 정답(true_label)과 정오(correct)를 함께 준다. 없는 필터 값은 빈 목록이다."""
    repo = PredictionRepository(session)
    wrong_flag = None if wrong is None else wrong == "true"
    items = repo.list_with_truth(model_version=model_version, inspection_id=inspection_id, wrong=wrong_flag,
                                 limit=limit, offset=offset)
    total = repo.count(model_version=model_version, inspection_id=inspection_id, wrong=wrong_flag)
    return PredictionList(items=[PredictionWithTruthOut.with_truth(i) for i in items], total=total)


@router.get("/stats", response_model=PredictionStatsOut, responses=_ERRORS)
def prediction_stats(
    session: SessionDep, _: CurrentUser,
    recent: Annotated[int, Query(ge=1, le=MAX_LIMIT, description="최근 N건 (정확도·재현율 계산 범위)")] = DEFAULT_RECENT,
    model_version: Annotated[Optional[str], Query(min_length=1, max_length=64)] = None,
) -> PredictionStatsOut:
    """판정 통계. ng_rate 는 판정이 NG인 비율이고, recent 는 최신 N건을 검사의 정답과 비교한 결과다."""
    return PredictionStatsOut.of(PredictionRepository(session).stats(recent=recent, model_version=model_version))
