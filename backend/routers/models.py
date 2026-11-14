"""모델 버전 조회: GET /models, /models/production, /models/{version}. 로그인한 누구나."""
import re
from typing import Annotated, Optional

from fastapi import APIRouter, Query

from backend.api_models import ErrorResponse, ModelList, ModelVersionOut
from backend.deps import CurrentUser, SessionDep
from backend.errors import ApiError
from backend.repository import ModelVersionRepository, Stage

router = APIRouter(prefix="/models", tags=["models"])

_ERRORS = {400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}}
_NOT_FOUND = {**_ERRORS, 404: {"model": ErrorResponse}}
_VERSION_NAME = re.compile(r"v[0-9]+")  # fullmatch 로만 쓴다 (숫자는 ASCII 만, 끝의 줄바꿈도 허용하지 않는다)


@router.get("", response_model=ModelList, responses=_ERRORS)
def list_models(session: SessionDep, _: CurrentUser,
                stage: Annotated[Optional[Stage], Query(description="이 단계의 버전만")] = None) -> ModelList:
    """모델 버전 목록, 최신 먼저 (생성 시각 내림차순)."""
    records = ModelVersionRepository(session).list()  # 저장소는 오래된 것부터 준다
    items = [ModelVersionOut.of(r) for r in reversed(records) if stage is None or r.stage is stage]
    return ModelList(items=items, total=len(items))


@router.get("/production", response_model=ModelVersionOut, responses=_NOT_FOUND)
def get_production(session: SessionDep, _: CurrentUser) -> ModelVersionOut:
    """현재 운영 중인 모델. 없으면 404 (/predict 의 503 과 문구는 같지만 상태 코드와 error 이름이 다르다)."""
    record = ModelVersionRepository(session).get_production()
    if record is None:
        raise ApiError(404, "운영 중인 모델이 없습니다")
    return ModelVersionOut.of(record)


@router.get("/{version}", response_model=ModelVersionOut, responses=_NOT_FOUND)
def get_model(version: str, session: SessionDep, _: CurrentUser) -> ModelVersionOut:
    """한 버전. v숫자 형식이 아니거나 없는 버전은 404 (형식이 틀리면 DB 를 읽지 않는다)."""
    record = ModelVersionRepository(session).get(version) if _VERSION_NAME.fullmatch(version) else None
    if record is None:
        raise ApiError(404, "모델 버전을 찾을 수 없습니다")
    return ModelVersionOut.of(record)
