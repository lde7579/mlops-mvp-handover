"""모델 승격·거절·롤백: 한 트랜잭션 안에서 stage 변경, 승인 요청, 배포 이력, 감사 기록을 함께 바꾼다 (계약 5~7장, C7).

- 이 모듈은 커밋하지 않는다. 호출하는 쪽의 session_scope() 가 커밋/롤백한다 (저장소와 같은 규칙).
  그래서 중간에 오류가 나면 이 함수가 한 일은 모두 되돌려지고, 반쯤 바뀐 상태가 남지 않는다.
- 쓰기 전에 검증을 모두 끝낸다 (승인 요청이 pending 인지, 후보가 staging 인지, 사유가 있는지 등).
- 순서가 중요하다: 기존 production 을 archived 로 먼저 내리고, 그다음 새 버전을 production 으로 올린다.
  production 은 최대 1개라는 DB 제약(uq_model_version_one_production)이 순서가 바뀌면 거부한다.
- 추론부 모델 핫리로드는 여기서 부르지 않는다. backend/deployment.py(DeployService)가 "새 모델 먼저 읽어 두기 -> 이 함수로 커밋 -> 모델 교체" 순서로 부른다.
- 권한(admin 만 승인·롤백)은 API 단계에서 막는다. 여기서는 role 을 감사 기록에 남기기만 한다.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from backend.database import ensure_aware
from backend.repository import (
    ApprovalRepository,
    ApprovalStatus,
    AuditAction,
    AuditLogRecord,
    AuditLogRepository,
    DeploymentAction,
    DeploymentRepository,
    InvalidTransitionError,
    ModelVersionRepository,
    NotFoundError,
    Role,
    Stage,
    TrainingRunRepository,
    _enum_value,
    _reason_required,
    _text,
)
from services.training.gate import GateResult

SYSTEM = "system"


@dataclass(frozen=True)
class DeployResult:
    """승격 또는 롤백의 결과. old_version 은 내려간 이전 production (없었으면 None)."""
    new_version: str
    old_version: Optional[str]
    approval_id: Optional[str] = None


@dataclass(frozen=True)
class SubmitResult:
    """게이트 결과를 적용한 결과. 통과하면 staging 과 approval_id, 미통과면 rejected 이고 approval_id 는 None."""
    version: str
    stage: Stage
    approval_id: Optional[str]


def record_audit(session: Session, *, actor: str, role, action, target: str, now: datetime) -> AuditLogRecord:
    """감사 기록 한 건. 4-C 가 핫리로드 실패를 롤백과 별개의 세션으로 남길 때 단독으로 부른다."""
    return AuditLogRepository(session).add(actor=actor, role=role, action=action, target=target, at=now)


def _check_actor(actor, role, now) -> None:
    _text("actor", actor)
    _enum_value(Role, "role", role)
    ensure_aware("now", now)


def _staging_candidate(session: Session, approval_id: str):
    """처리할 수 있는 (pending 승인 요청, staging 후보) 를 돌려준다. 아니면 NotFoundError / InvalidTransitionError."""
    approval = ApprovalRepository(session).get(approval_id)
    if approval is None:
        raise NotFoundError(f"없는 approval_id 입니다: {approval_id}")
    if approval.status is not ApprovalStatus.PENDING:
        raise InvalidTransitionError(f"{approval_id}: 이미 처리된 승인 요청입니다 ({approval.status.value})")
    model = ModelVersionRepository(session).get(approval.model_version)
    if model is None:
        raise NotFoundError(f"없는 버전입니다: {approval.model_version}")
    if model.stage is not Stage.STAGING:
        raise InvalidTransitionError(f"{model.version}: staging 이 아니라 {model.stage.value} 입니다")
    return approval, model


def submit_candidate(session: Session, version: str, gate: GateResult, *, now: datetime,
                     training_run_id: Optional[str] = None) -> SubmitResult:
    """후보(candidate)에 게이트 결과를 적용한다.

    통과: staging 으로 올리고 pending 승인 요청(요청자 system)을 만든다, 감사 기록 gate_pass.
    미통과: rejected 로 바꾼다. 승인 요청은 만들지 않는다 (e2e X1), 감사 기록 gate_fail.
    training_run_id 를 주면 그 실행 기록의 gate_result 에 baseline_version 을 포함한 결과를 남긴다.
    """
    ensure_aware("now", now)
    models, runs = ModelVersionRepository(session), TrainingRunRepository(session)
    model = models.get(version)
    if model is None:
        raise NotFoundError(f"없는 버전입니다: {version}")
    if model.stage is not Stage.CANDIDATE:
        raise InvalidTransitionError(f"{version}: candidate 가 아니라 {model.stage.value} 입니다")
    if training_run_id is not None and runs.get(training_run_id) is None:
        raise NotFoundError(f"없는 run_id 입니다: {training_run_id}")

    approval_id = None
    if gate.passed:
        approval_id = ApprovalRepository(session).add_pending(version, requested_by=SYSTEM, created_at=now).approval_id
        models.transition(version, Stage.STAGING)
        action = AuditAction.GATE_PASS
    else:
        models.transition(version, Stage.REJECTED)
        action = AuditAction.GATE_FAIL
    if training_run_id is not None:
        runs.update(training_run_id, gate_result=gate.to_dict())
    record_audit(session, actor=SYSTEM, role=Role.SYSTEM, action=action, target=version, now=now)
    return SubmitResult(version=version, stage=Stage.STAGING if gate.passed else Stage.REJECTED, approval_id=approval_id)


def promote(session: Session, approval_id: str, *, actor: str, role, now: datetime,
            reason: Optional[str] = None) -> DeployResult:
    """승인: pending 요청의 후보(staging)를 production 으로 올린다.

    순서: ① 기존 production -> archived ② 후보 -> production (approved_by·approved_at 기록)
    ③ 승인 요청 approved ④ deployment(deploy) ⑤ 감사 기록 approve.
    이미 처리된 요청이나 staging 이 아닌 후보는 InvalidTransitionError, 없는 요청은 NotFoundError (쓰기 전에 확인한다).
    """
    _check_actor(actor, role, now)
    _, model = _staging_candidate(session, approval_id)

    models = ModelVersionRepository(session)
    old = models.archive_production()
    models.transition(model.version, Stage.PRODUCTION, approved_by=actor, approved_at=now)
    ApprovalRepository(session).mark_handled(approval_id, status=ApprovalStatus.APPROVED, handled_by=actor,
                                             handled_at=now, reason=reason)
    DeploymentRepository(session).add(model.version, DeploymentAction.DEPLOY, actor=actor, at=now, reason=reason)
    record_audit(session, actor=actor, role=role, action=AuditAction.APPROVE, target=model.version, now=now)
    return DeployResult(new_version=model.version, old_version=None if old is None else old.version, approval_id=approval_id)


def reject(session: Session, approval_id: str, *, actor: str, role, reason: str, now: datetime) -> SubmitResult:
    """거절: pending 요청의 후보(staging)를 rejected 로 바꾼다. 사유가 없으면 ValueError (e2e X5)."""
    _check_actor(actor, role, now)
    _reason_required(reason)
    _, model = _staging_candidate(session, approval_id)

    ModelVersionRepository(session).transition(model.version, Stage.REJECTED)
    ApprovalRepository(session).mark_handled(approval_id, status=ApprovalStatus.REJECTED, handled_by=actor,
                                             handled_at=now, reason=reason)
    record_audit(session, actor=actor, role=role, action=AuditAction.REJECT, target=model.version, now=now)
    return SubmitResult(version=model.version, stage=Stage.REJECTED, approval_id=approval_id)


def rollback(session: Session, version: str, *, actor: str, role, reason: str, now: datetime) -> DeployResult:
    """롤백: archived 버전을 production 으로 되돌린다. 사유가 없으면 ValueError (e2e X5).

    순서는 승격과 같다 (현재 production -> archived 먼저). approved_by·approved_at 은 롤백한 사람과 시각으로 바뀌고
    ("마지막으로 production 에 올린 사람과 시각", C7), 승인 요청 원본은 건드리지 않는다.
    """
    _check_actor(actor, role, now)
    _reason_required(reason)
    models = ModelVersionRepository(session)
    target = models.get(version)
    if target is None:
        raise NotFoundError(f"없는 버전입니다: {version}")
    if target.stage is not Stage.ARCHIVED:
        raise InvalidTransitionError(f"{version}: archived 가 아니라 {target.stage.value} 라서 롤백할 수 없습니다")

    old = models.archive_production()
    models.transition(version, Stage.PRODUCTION, approved_by=actor, approved_at=now)
    DeploymentRepository(session).add(version, DeploymentAction.ROLLBACK, actor=actor, at=now, reason=reason)
    record_audit(session, actor=actor, role=role, action=AuditAction.ROLLBACK, target=version, now=now)
    return DeployResult(new_version=version, old_version=None if old is None else old.version)
