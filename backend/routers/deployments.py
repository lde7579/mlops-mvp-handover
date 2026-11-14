"""배포 이력 GET /deployments (로그인한 누구나)와 롤백 POST /models/{version}/rollback (관리자만). api-v1.md A20, A21."""
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query

from backend.api_models import DeploymentList, DeploymentOut, ErrorResponse, ReasonRequest, ReloadOut, RollbackOut
from backend.deployment import to_api_error
from backend.deps import CurrentUser, DeployDep, SessionDep, User, require_role
from backend.inference import ModelUnavailableError
from backend.repository import MAX_LIMIT, ConflictError, DeploymentAction, DeploymentRepository, NotFoundError
from backend.security import Role

router = APIRouter(tags=["deployments"])

_ERRORS = {400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}}


@router.get("/deployments", response_model=DeploymentList, responses=_ERRORS)
def list_deployments(session: SessionDep, _: CurrentUser,
                     version: Annotated[Optional[str], Query(description="이 모델 버전의 기록만")] = None,
                     action: Annotated[Optional[DeploymentAction], Query(description="deploy 또는 rollback")] = None,
                     limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = 50,
                     offset: Annotated[int, Query(ge=0)] = 0) -> DeploymentList:
    """배포·롤백 이력, 최신 먼저. total 은 필터를 적용한 전체 건수다. 없는 version 은 빈 목록."""
    deployments = DeploymentRepository(session)
    items = [DeploymentOut.of(d) for d in deployments.list(version=version, action=action, limit=limit, offset=offset)]
    return DeploymentList(items=items, total=deployments.count(version=version, action=action))


@router.post("/models/{version}/rollback", response_model=RollbackOut,
             responses={**{code: {"model": ErrorResponse} for code in (400, 401, 403, 404, 409, 503)}})
def rollback_model(version: str, body: ReasonRequest, deploy: DeployDep,
                   user: Annotated[User, Depends(require_role(Role.ADMIN))]) -> RollbackOut:
    """롤백 (관리자만, 사유 필수). archived 버전을 production 으로 되돌리고 추론부 모델을 바꾼 뒤에 응답한다.

    그 버전의 모델 파일이 없거나 불러오지 못하면 503 이고 아무것도 바뀌지 않는다.
    """
    try:
        r = deploy.rollback(version, actor=user.username, role=user.role, reason=body.reason)
    except (NotFoundError, ConflictError, ModelUnavailableError) as e:
        raise to_api_error(e) from None
    return RollbackOut(deployment=DeploymentOut.of(r.deployment), new_version=r.new_version, old_version=r.old_version,
                       reload=ReloadOut(loaded_version=r.loaded_version, elapsed_ms=r.elapsed_ms))
