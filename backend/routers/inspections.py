"""검사 조회: 목록, 상세, 이미지 (로그인한 누구나)."""
from datetime import datetime
from typing import Annotated, Optional

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from backend.api_models import ErrorResponse, InspectionDetail, InspectionList, InspectionOut, PredictionOut
from backend.deps import CurrentUser, SessionDep, SettingsDep
from backend.errors import ApiError
from backend.images import resolve_image
from backend.repository import MAX_LIMIT, InspectionRepository, PredictionRepository
from common.schemas import DataSplit, Scenario

router = APIRouter(prefix="/inspections", tags=["inspections"])

_ERRORS = {400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}}
_NOT_FOUND = "검사를 찾을 수 없습니다"


def parse_time(name: str, value: Optional[str]) -> Optional[datetime]:
    """ISO 8601 시각 (시간대 필수). 숫자만 있는 값이나 날짜만 있는 값은 받지 않는다."""
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        parsed = None
    if parsed is None or parsed.utcoffset() is None:
        raise ApiError(400, f"{name}: 시간대가 있는 ISO 8601 시각이어야 합니다 (예: 2026-10-18T09:00:00+09:00, '+'는 %2B로 보내세요)")
    return parsed


@router.get("", response_model=InspectionList, responses=_ERRORS)
def list_inspections(
    session: SessionDep, _: CurrentUser,
    scenario: Annotated[Optional[Scenario], Query(description="normal 또는 drift")] = None,
    data_split: Annotated[Optional[DataSplit], Query(description="train, pool, test_normal, test_drift")] = None,
    since: Annotated[Optional[str], Query(description="이 시각 이후 (포함). 시간대 필수")] = None,
    until: Annotated[Optional[str], Query(description="이 시각 이전 (미포함). 시간대 필수")] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> InspectionList:
    """검사 목록, 최신 먼저. total 은 limit·offset 과 무관한 필터 결과 전체 건수다."""
    start, end = parse_time("since", since), parse_time("until", until)
    if start is not None and end is not None and start > end:
        raise ApiError(400, "since 는 until 보다 늦을 수 없습니다")
    repo = InspectionRepository(session)
    filters = {"scenario": scenario, "data_split": data_split, "since": start, "until": end}
    items = repo.list(**filters, limit=limit, offset=offset)
    return InspectionList(items=[InspectionOut.of(i) for i in items], total=repo.count(**filters))


@router.get("/{inspection_id}", response_model=InspectionDetail, responses={**_ERRORS, 404: {"model": ErrorResponse}})
def get_inspection(inspection_id: str, session: SessionDep, _: CurrentUser) -> InspectionDetail:
    inspection = InspectionRepository(session).get(inspection_id)
    if inspection is None:
        raise ApiError(404, _NOT_FOUND)
    predictions = PredictionRepository(session).list(inspection_id=inspection_id, limit=MAX_LIMIT)
    return InspectionDetail(**InspectionOut.of(inspection).model_dump(), predictions=[PredictionOut.of(p) for p in predictions])


@router.get("/{inspection_id}/image", response_class=FileResponse,
            responses={**_ERRORS, 404: {"model": ErrorResponse}, 200: {"content": {"image/png": {}}}})
def get_image(inspection_id: str, session: SessionDep, settings: SettingsDep, _: CurrentUser) -> FileResponse:
    inspection = InspectionRepository(session).get(inspection_id)
    if inspection is None:
        raise ApiError(404, _NOT_FOUND)
    path = resolve_image(settings.root, inspection.image_path)
    if path is None:  # 없는 파일, 허용되지 않는 경로 모두 같은 응답 (이유는 로그에만)
        raise ApiError(404, "이미지를 찾을 수 없습니다")
    return FileResponse(path, media_type="image/png")
