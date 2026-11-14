"""승인·거절·롤백 흐름: DeployService (api-v1.md A18~A20, 4장).

승인의 순서 (모델을 읽는 동안 DB 쓰기 잠금을 쥐지 않고, 읽기에 실패해도 DB 가 바뀌지 않게 한다):
  ① 읽기 전용 세션으로 확인: 없는 요청 404, 이미 처리됨·후보가 staging 아님·기준 모델 바뀜 409. 여기서 걸리면 모델을 읽지 않는다.
  ② inference.prepare 로 새 모델을 별도로 읽어 둔다. 실패하면 아무것도 커밋하지 않고 별도 세션으로 실패 기록(approve_fail) 후 503.
  ③ 쓰기 트랜잭션에서 다시 확인하고(promote 가 검증한다) stage 변경·승인 요청·배포 이력·감사 기록을 한 번에 커밋한다.
     실패하면 읽어 둔 모델을 버린다.
  ④ 커밋이 성공한 뒤 inference.activate 로 모델을 바꾼다. 그 뒤에 응답한다.
- 세 작업은 락 하나로 한 번에 하나씩만 실행한다 (두 관리자가 동시에 눌러도 흐름이 섞이지 않는다). /predict 는 이 락을 쓰지 않는다.
- 세션은 SessionDep 대신 session_factory 로 직접 연다: 핸들러가 끝나야 커밋되는 방식으로는 "커밋 -> 모델 교체" 순서를 보장할 수 없다.
- 오류는 저장소의 예외로 돌려준다: NotFoundError(404), ConflictError(409, InvalidTransitionError 포함), ModelUnavailableError(503).
"""
import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from backend.database import session_scope
from backend.errors import ApiError
from backend.inference import ModelUnavailableError
from backend.promotion import _staging_candidate, promote, record_audit
from backend.promotion import reject as promotion_reject
from backend.promotion import rollback as promotion_rollback
from backend.repository import (
    ApprovalRecord,
    ApprovalRepository,
    AuditAction,
    ConflictError,
    DeploymentAction,
    DeploymentRecord,
    DeploymentRepository,
    InvalidTransitionError,
    ModelVersionRepository,
    NotFoundError,
    Stage,
    TrainingRunRepository,
)

log = logging.getLogger("backend")

_APPROVAL_ID = re.compile(r"APR-[0-9]+")  # fullmatch 로만 쓴다 (숫자는 ASCII 만, 끝의 줄바꿈도 허용하지 않는다)
_VERSION_NAME = re.compile(r"v[0-9]+")
APPROVAL_NOT_FOUND = "승인 요청을 찾을 수 없습니다"
VERSION_NOT_FOUND = "모델 버전을 찾을 수 없습니다"


def to_api_error(error: Exception) -> ApiError:
    """흐름이 던지는 예외를 응답 오류로 바꾼다: 없음 404, 상태 충돌 409, 모델을 불러오지 못함 503. 그 밖의 예외는 바꾸지 않는다(500)."""
    if isinstance(error, NotFoundError):
        return ApiError(404, str(error))
    if isinstance(error, ConflictError):
        return ApiError(409, str(error))
    if isinstance(error, ModelUnavailableError):
        return ApiError(503, str(error))
    raise error


@dataclass(frozen=True)
class RejectResult:
    approval: ApprovalRecord
    version: str


@dataclass(frozen=True)
class ApproveResult:
    approval: ApprovalRecord
    deployment: DeploymentRecord
    new_version: str
    old_version: Optional[str]
    loaded_version: str
    elapsed_ms: int


@dataclass(frozen=True)
class RollbackResult:
    deployment: DeploymentRecord
    new_version: str
    old_version: Optional[str]
    loaded_version: str
    elapsed_ms: int


def _check_baseline(session, version: str) -> Optional[str]:
    """게이트가 비교한 기준 모델이 지금 production 과 같은지 확인한다 (둘 다 없으면 같다). 현재 production 버전을 돌려준다.

    후보의 학습 기록이 없으면 확인할 근거가 없으므로 넘어간다. 다르면 ConflictError (api-v1.md 4장 "기준 모델 검사").
    """
    production = ModelVersionRepository(session).get_production()
    current = None if production is None else production.version
    run = TrainingRunRepository(session).get_by_result_version(version)
    if run is not None and run.gate_result is not None:
        baseline = run.gate_result.get("baseline_version")
        if baseline != current:
            raise ConflictError(f"게이트를 통과할 때의 기준 모델({baseline or '없음'})이 이제 운영 모델({current or '없음'})과 다릅니다. "
                                "이 요청을 거절하고 다시 재학습해 주세요")
    return current


class DeployService:
    def __init__(self, session_factory, inference, *, clock: Optional[Callable[[], datetime]] = None):
        self._factory = session_factory
        self._inference = inference
        self._clock = clock if clock is not None else (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()

    def reject(self, approval_id: str, *, actor: str, role, reason: str) -> RejectResult:
        """거절: staging 후보를 rejected 로 바꾸고 승인 요청을 rejected 로 처리한다. 추론부 모델은 건드리지 않는다."""
        if not _APPROVAL_ID.fullmatch(approval_id):
            raise NotFoundError(APPROVAL_NOT_FOUND)
        with self._lock, session_scope(self._factory) as session:
            if ApprovalRepository(session).get(approval_id) is None:
                raise NotFoundError(APPROVAL_NOT_FOUND)
            result = promotion_reject(session, approval_id, actor=actor, role=role, reason=reason, now=self._clock())
            return RejectResult(approval=ApprovalRepository(session).get(approval_id), version=result.version)

    def approve(self, approval_id: str, *, actor: str, role, reason: Optional[str] = None) -> ApproveResult:
        """승인: 후보를 production 으로 올리고 추론부 모델을 바꾼다. 위 설명의 ①~④ 순서다."""
        if not _APPROVAL_ID.fullmatch(approval_id):
            raise NotFoundError(APPROVAL_NOT_FOUND)
        with self._lock:
            with session_scope(self._factory) as session:  # ① 읽기 전용 확인 (쓰지 않는다)
                if ApprovalRepository(session).get(approval_id) is None:
                    raise NotFoundError(APPROVAL_NOT_FOUND)
                _, candidate = _staging_candidate(session, approval_id)
                version = candidate.version
                current = _check_baseline(session, version)
            prepared, elapsed_ms = self._prepare(version, AuditAction.APPROVE_FAIL, "승인", actor, role, current)  # ② 읽어 둔다
            try:  # ③ 짧은 쓰기 트랜잭션 (다시 확인하고 한 번에 커밋)
                with session_scope(self._factory) as session:
                    _check_baseline(session, version)
                    deployed = promote(session, approval_id, actor=actor, role=role, now=self._clock(), reason=reason)
                    approval = ApprovalRepository(session).get(approval_id)
                    deployment = DeploymentRepository(session).list(version=version, action=DeploymentAction.DEPLOY, limit=1)[0]
            except BaseException:
                self._inference.discard(prepared)
                raise
            self._activate(prepared)  # ④ 커밋이 끝난 뒤에만 바꾼다
            return ApproveResult(approval=approval, deployment=deployment, new_version=deployed.new_version,
                                 old_version=deployed.old_version, loaded_version=version, elapsed_ms=elapsed_ms)

    def rollback(self, version: str, *, actor: str, role, reason: str) -> RollbackResult:
        """롤백: archived 버전을 production 으로 되돌리고 추론부 모델을 바꾼다. 승인과 같은 ①~④ 순서다 (사유는 필수)."""
        if not _VERSION_NAME.fullmatch(version):
            raise NotFoundError(VERSION_NOT_FOUND)
        with self._lock:
            with session_scope(self._factory) as session:  # ① 읽기 전용 확인 (쓰지 않는다)
                models = ModelVersionRepository(session)
                target = models.get(version)
                if target is None:
                    raise NotFoundError(VERSION_NOT_FOUND)
                if target.stage is not Stage.ARCHIVED:
                    raise InvalidTransitionError(f"{version}: archived 가 아니라 {target.stage.value} 라서 롤백할 수 없습니다")
                production = models.get_production()
                current = None if production is None else production.version
            prepared, elapsed_ms = self._prepare(version, AuditAction.ROLLBACK_FAIL, "롤백", actor, role, current)  # ② 읽어 둔다
            try:  # ③ 짧은 쓰기 트랜잭션
                with session_scope(self._factory) as session:
                    rolled = promotion_rollback(session, version, actor=actor, role=role, reason=reason, now=self._clock())
                    deployment = DeploymentRepository(session).list(version=version, action=DeploymentAction.ROLLBACK, limit=1)[0]
            except BaseException:
                self._inference.discard(prepared)
                raise
            self._activate(prepared)  # ④ 커밋이 끝난 뒤에만 바꾼다
            return RollbackResult(deployment=deployment, new_version=rolled.new_version, old_version=rolled.old_version,
                                  loaded_version=version, elapsed_ms=elapsed_ms)

    def _prepare(self, version: str, fail_action: AuditAction, verb: str, actor: str, role, current: Optional[str]):
        """② 새 모델을 읽어 둔다. 실패하면 감사 기록(별도 세션)을 남기고 ModelUnavailableError. (읽어 둔 모델, 걸린 ms)를 돌려준다."""
        started = time.perf_counter()
        try:
            prepared = self._inference.prepare(version)
        except ModelUnavailableError:
            self._record_failure(fail_action, version, actor, role)
            keep = f"기존 운영 모델 {current} 을 계속 사용합니다" if current else "운영 모델은 바뀌지 않았습니다"
            noun = "새 모델" if verb == "승인" else "모델"  # api-v1.md A18, A20 의 메시지
            raise ModelUnavailableError(f"{noun} {version} 을 불러오지 못해 {verb}하지 않았습니다. {keep}") from None
        return prepared, round((time.perf_counter() - started) * 1000)

    def _activate(self, prepared) -> None:
        """교체가 실행되지 못해도 이미 커밋됐으므로 응답을 막지 않는다: 다음 판정이 DB 의 production 을 보고 읽어 둔 모델을 쓴다."""
        try:
            self._inference.activate(prepared)
        except Exception:
            log.exception("커밋은 끝났지만 추론부 모델 교체를 실행하지 못했습니다. 다음 판정 때 DB 의 운영 모델로 바뀝니다")

    def _record_failure(self, action: AuditAction, target: str, actor: str, role) -> None:
        """모델을 불러오지 못한 일을 감사 기록에 남긴다. 승인·롤백 트랜잭션과 별개의 세션이라 되돌려지지 않는다."""
        try:
            with session_scope(self._factory) as session:
                record_audit(session, actor=actor, role=role, action=action, target=target, now=self._clock())
        except Exception:
            log.exception("%s 감사 기록을 남기지 못했습니다 (%s)", action.value, target)
