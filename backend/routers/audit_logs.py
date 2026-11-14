"""감사 로그: GET /audit-logs (관리자만). api-v1.md A22."""
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query

from backend.api_models import AuditLogList, AuditLogOut, ErrorResponse
from backend.deps import SessionDep, User, require_role
from backend.repository import MAX_LIMIT, AuditAction, AuditLogRepository
from backend.security import Role

router = APIRouter(tags=["audit"])

_ERRORS = {code: {"model": ErrorResponse} for code in (400, 401, 403)}


@router.get("/audit-logs", response_model=AuditLogList, responses=_ERRORS)
def list_audit_logs(session: SessionDep, _: Annotated[User, Depends(require_role(Role.ADMIN))],
                    action: Annotated[Optional[AuditAction], Query(description="이 action 의 기록만")] = None,
                    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = 50,
                    offset: Annotated[int, Query(ge=0)] = 0) -> AuditLogList:
    """감사 로그, 최신 먼저. total 은 필터를 적용한 전체 건수다."""
    logs = AuditLogRepository(session)
    items = [AuditLogOut.of(a) for a in logs.list(action=action, limit=limit, offset=offset)]
    return AuditLogList(items=items, total=logs.count(action=action))
