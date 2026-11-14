"""모니터링 조회: GET /monitoring/performance (구간 추이, 최근 N건), GET /monitoring/drift (성능 저하 경고, 입력 분포). 로그인한 누구나.

N=30·정확도 0.95는 5주차 측정으로 확정한 값이다 (응답의 confirmed=true). api-v1.md A24·A25.
"""
from typing import Annotated, Optional

from fastapi import APIRouter, Query

from backend.api_models import DriftOut, ErrorResponse, PerformanceOut
from backend.deps import CurrentUser, MonitoringDep, SessionDep
from backend.errors import ApiError
from backend.monitoring import RECENT_N, MonitoringInputError
from backend.repository import MAX_LIMIT
from backend.routers.inspections import parse_time

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

_ERRORS = {400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}}
_RECENT = Query(ge=1, le=MAX_LIMIT, description="최근 N건 (기본 30, 확정값)")
_VERSION = Query(min_length=1, max_length=64, description="이 모델 버전의 판정만 (생략하면 recent 는 현재 production)")


@router.get("/performance", response_model=PerformanceOut, responses=_ERRORS)
def performance(
    session: SessionDep, monitoring: MonitoringDep, _: CurrentUser,
    since: Annotated[Optional[str], Query(description="구간 시작 (포함). 시간대 필수. 기본: until 의 1시간 전")] = None,
    until: Annotated[Optional[str], Query(description="구간 끝 (미포함). 시간대 필수. 기본: 지금")] = None,
    bucket_seconds: Annotated[int, Query(ge=10, le=3600, description="구간 길이(초)")] = 30,
    recent: Annotated[int, _RECENT] = RECENT_N,
    model_version: Annotated[Optional[str], _VERSION] = None,
) -> PerformanceOut:
    """시간대별 정확도·재현율 추이(points)와 최근 N건 정확도(recent). points 는 (구간, 모델 버전)마다 한 줄이다."""
    try:
        result = monitoring.performance(session, since=parse_time("since", since), until=parse_time("until", until),
                                        bucket_seconds=bucket_seconds, recent=recent, model_version=model_version)
    except MonitoringInputError as e:
        raise ApiError(400, str(e)) from None
    return PerformanceOut.of(result)


@router.get("/drift", response_model=DriftOut, responses=_ERRORS)
def drift(
    session: SessionDep, monitoring: MonitoringDep, _: CurrentUser,
    recent: Annotated[int, _RECENT] = RECENT_N,
    model_version: Annotated[Optional[str], _VERSION] = None,
) -> DriftOut:
    """성능 저하 경고(performance_warning)와 입력 분포 변화(input_distribution, 값만). 경고는 재학습을 시작하지 않는다."""
    return DriftOut.of(monitoring.drift(session, recent=recent, model_version=model_version))
