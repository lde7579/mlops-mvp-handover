"""승인 요청: GET /approvals (로그인한 누구나). 승인·거절은 같은 파일에 이어서 붙는다 (관리자만). api-v1.md A17~A19."""
from typing import Annotated, Optional

from fastapi import APIRouter, Body, Depends, Query

from backend.api_models import (
    ApprovalList,
    ApprovalOut,
    ApproveOut,
    ApproveRequest,
    DeploymentOut,
    ErrorResponse,
    ReasonRequest,
    RejectOut,
    ReloadOut,
)
from backend.deployment import to_api_error
from backend.deps import CurrentUser, DeployDep, SessionDep, User, require_role
from backend.inference import ModelUnavailableError
from backend.repository import MAX_LIMIT, ApprovalRepository, ApprovalStatus, ConflictError, NotFoundError
from backend.security import Role

router = APIRouter(prefix="/approvals", tags=["approvals"])

_ERRORS = {400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}}


@router.get("", response_model=ApprovalList, responses=_ERRORS)
def list_approvals(session: SessionDep, _: CurrentUser,
                   status: Annotated[Optional[ApprovalStatus], Query(description="이 상태의 요청만")] = None,
                   limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = 50,
                   offset: Annotated[int, Query(ge=0)] = 0) -> ApprovalList:
    """승인 요청 목록, 최신 먼저. total 은 필터를 적용한 전체 건수다."""
    approvals = ApprovalRepository(session)
    items = [ApprovalOut.of(a) for a in approvals.list(status=status, limit=limit, offset=offset)]
    return ApprovalList(items=items, total=approvals.count(status=status))


_ADMIN = Annotated[User, Depends(require_role(Role.ADMIN))]
_WRITE_ERRORS = {code: {"model": ErrorResponse} for code in (400, 401, 403, 404, 409)}


@router.post("/{approval_id}/approve", response_model=ApproveOut, responses={**_WRITE_ERRORS, 503: {"model": ErrorResponse}})
def approve_approval(approval_id: str, deploy: DeployDep, user: _ADMIN,
                     body: Annotated[ApproveRequest, Body()] = ApproveRequest()) -> ApproveOut:
    """승인 (관리자만, 사유는 선택). 후보가 production 이 되고 추론부 모델이 바뀐 뒤에 응답한다.

    새 모델을 불러오지 못하면 503 이고 아무것도 바뀌지 않는다 (승인 요청은 pending 으로 남는다).
    """
    try:
        r = deploy.approve(approval_id, actor=user.username, role=user.role, reason=body.reason)
    except (NotFoundError, ConflictError, ModelUnavailableError) as e:
        raise to_api_error(e) from None
    return ApproveOut(approval=ApprovalOut.of(r.approval), deployment=DeploymentOut.of(r.deployment), new_version=r.new_version,
                      old_version=r.old_version, reload=ReloadOut(loaded_version=r.loaded_version, elapsed_ms=r.elapsed_ms))


@router.post("/{approval_id}/reject", response_model=RejectOut, responses=_WRITE_ERRORS)
def reject_approval(approval_id: str, body: ReasonRequest, deploy: DeployDep, user: _ADMIN) -> RejectOut:
    """거절 (관리자만, 사유 필수). 후보는 rejected, 승인 요청은 rejected 가 된다. 이미 처리된 요청은 409."""
    try:
        result = deploy.reject(approval_id, actor=user.username, role=user.role, reason=body.reason)
    except (NotFoundError, ConflictError) as e:
        raise to_api_error(e) from None
    return RejectOut(approval=ApprovalOut.of(result.approval), version=result.version, stage="rejected")
