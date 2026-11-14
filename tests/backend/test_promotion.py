"""backend/promotion.py: 게이트 적용, 승인, 거절, 롤백 (계약 5~7장 C7, e2e X1·X4·X5). 메모리 DB, torch·학습 없음."""
import pytest
from sqlalchemy import event, select

from backend.database import session_scope
from backend.models import ApprovalRequestRow, AuditLogRow, DeploymentRow, ModelVersionRow
from backend.promotion import SubmitResult, promote, record_audit, reject, rollback, submit_candidate
from backend.repository import (
    ApprovalRepository,
    ApprovalStatus,
    AuditAction,
    AuditLogRepository,
    ConflictError,
    DeploymentRepository,
    InvalidTransitionError,
    ModelVersionRepository,
    NotFoundError,
    Stage,
    TrainingRunRepository,
)
from services.training.gate import evaluate_gate
from tests.backend.helpers import at, make_metrics

GOOD = make_metrics("v2", created_minutes=1, drift={"accuracy": 0.97, "recall": 1.0, "precision": 0.9, "f1": 0.95,
                                                    "confusion_matrix": [[385, 15], [0, 100]], "n": 500})
BAD = make_metrics("v3", created_minutes=2, drift={"accuracy": 0.5, "recall": 0.5, "precision": 0.5, "f1": 0.5,
                                                   "confusion_matrix": [[200, 200], [50, 50]], "n": 500})


def stage_of(session, version):
    return ModelVersionRepository(session).get(version).stage


@pytest.fixture
def world(session):
    """v1 production, v2 staging(승인 대기 APR-000001), v3 candidate."""
    models = ModelVersionRepository(session)
    models.register(make_metrics("v1"), stage=Stage.PRODUCTION)
    models.register(GOOD, stage=Stage.STAGING)
    models.register(BAD, stage=Stage.CANDIDATE)
    ApprovalRepository(session).add_pending("v2", requested_by="system", created_at=at(1))
    session.commit()
    return session


# ---------- 게이트 적용 (candidate -> staging / rejected) ----------

def test_submit_passing_candidate_goes_to_staging_with_a_pending_request(session):
    models = ModelVersionRepository(session)
    models.register(make_metrics("v1"), stage=Stage.PRODUCTION)
    models.register(GOOD)
    run = TrainingRunRepository(session).add("manual")
    gate = evaluate_gate(models.get("v1").metrics, models.get("v2").metrics)
    result = submit_candidate(session, "v2", gate, now=at(5), training_run_id=run.run_id)
    assert result == SubmitResult("v2", Stage.STAGING, "APR-000001") and stage_of(session, "v2") is Stage.STAGING
    approval = ApprovalRepository(session).get("APR-000001")
    assert (approval.model_version, approval.requested_by, approval.status) == ("v2", "system", ApprovalStatus.PENDING)
    log = AuditLogRepository(session).list()[0]
    assert (log.action, log.actor, log.role.value, log.target, log.at) == (AuditAction.GATE_PASS, "system", "system", "v2", at(5))
    saved = TrainingRunRepository(session).get(run.run_id).gate_result
    assert saved["baseline_version"] == "v1" and saved["passed"] is True and len(saved["checks"]) == 3


def test_submit_failing_candidate_is_rejected_without_an_approval_request(session):  # e2e X1
    models = ModelVersionRepository(session)
    models.register(make_metrics("v1"), stage=Stage.PRODUCTION)
    models.register(BAD)
    run = TrainingRunRepository(session).add("manual")
    gate = evaluate_gate(models.get("v1").metrics, models.get("v3").metrics)
    result = submit_candidate(session, "v3", gate, now=at(5), training_run_id=run.run_id)
    assert result == SubmitResult("v3", Stage.REJECTED, None) and stage_of(session, "v3") is Stage.REJECTED
    assert ApprovalRepository(session).count() == 0
    assert AuditLogRepository(session).list()[0].action is AuditAction.GATE_FAIL
    saved = TrainingRunRepository(session).get(run.run_id).gate_result
    assert saved["passed"] is False and saved["baseline_version"] == "v1" and [c["passed"] for c in saved["checks"]] == [False, False, True]  # G3(정상 정확도)은 BAD 도 1.0 이라 통과
    assert models.get_production().version == "v1"


def test_first_model_passes_the_gate_and_records_no_baseline(session):
    models = ModelVersionRepository(session)
    models.register(make_metrics("v1"))
    run = TrainingRunRepository(session).add("manual")
    result = submit_candidate(session, "v1", evaluate_gate(None, models.get("v1").metrics), now=at(1), training_run_id=run.run_id)
    assert result.stage is Stage.STAGING  # 게이트는 면제지만 사람 승인은 여전히 필요하다
    assert TrainingRunRepository(session).get(run.run_id).gate_result == {"baseline_version": None, "passed": True, "checks": []}


def test_submit_rejects_wrong_state_and_unknown_targets(world):
    gate = evaluate_gate(None, GOOD)
    with pytest.raises(InvalidTransitionError):
        submit_candidate(world, "v2", gate, now=at(5))  # 이미 staging
    with pytest.raises(NotFoundError):
        submit_candidate(world, "v9", gate, now=at(5))
    with pytest.raises(NotFoundError):
        submit_candidate(world, "v3", gate, now=at(5), training_run_id="RUN-000009")
    assert stage_of(world, "v3") is Stage.CANDIDATE and ApprovalRepository(world).count() == 1


# ---------- 승인 ----------

def test_promote_archives_old_production_then_raises_the_candidate(world):
    statements = []

    @event.listens_for(world.get_bind(), "before_cursor_execute")
    def record(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("UPDATE model_version"):
            statements.append(parameters[0])  # 바꾸는 stage 값 (UPDATE model_version SET stage=? ...)

    try:
        result = promote(world, "APR-000001", actor="admin", role="admin", now=at(10), reason="수치 확인")
    finally:
        event.remove(world.get_bind(), "before_cursor_execute", record)
    assert statements == ["archived", "production"]  # 순서: 기존 production 을 먼저 내린다
    assert (result.new_version, result.old_version, result.approval_id) == ("v2", "v1", "APR-000001")
    models = ModelVersionRepository(world)
    assert (models.get("v1").stage, models.get("v2").stage) == (Stage.ARCHIVED, Stage.PRODUCTION)
    assert models.get_production().version == "v2"
    v2 = models.get("v2")
    assert (v2.approved_by, v2.approved_at) == ("admin", at(10))
    approval = ApprovalRepository(world).get("APR-000001")
    assert (approval.status, approval.handled_by, approval.handled_at, approval.reason) == (ApprovalStatus.APPROVED, "admin", at(10), "수치 확인")
    deployment = DeploymentRepository(world).list()
    assert len(deployment) == 1 and (deployment[0].version, deployment[0].action.value, deployment[0].actor, deployment[0].at, deployment[0].reason) == (
        "v2", "deploy", "admin", at(10), "수치 확인")
    log = AuditLogRepository(world).list()[0]
    assert (log.action, log.actor, log.role.value, log.target) == (AuditAction.APPROVE, "admin", "admin", "v2")


def test_promote_without_a_current_production_just_raises_the_candidate(session):
    models = ModelVersionRepository(session)
    models.register(make_metrics("v1"), stage=Stage.STAGING)
    ApprovalRepository(session).add_pending("v1", requested_by="system", created_at=at(1))
    result = promote(session, "APR-000001", actor="admin", role="admin", now=at(2))
    assert result.old_version is None and models.get_production().version == "v1"
    assert DeploymentRepository(session).list()[0].reason is None  # 승인 사유는 선택


def test_second_approval_of_the_same_request_is_a_conflict_and_changes_nothing(world):  # e2e X4
    promote(world, "APR-000001", actor="admin", role="admin", now=at(10))
    before = (len(DeploymentRepository(world).list()), len(AuditLogRepository(world).list()))
    with pytest.raises(InvalidTransitionError):
        promote(world, "APR-000001", actor="admin", role="admin", now=at(11))
    with pytest.raises(InvalidTransitionError):
        reject(world, "APR-000001", actor="admin", role="admin", reason="늦었다", now=at(11))
    assert (len(DeploymentRepository(world).list()), len(AuditLogRepository(world).list())) == before
    assert ModelVersionRepository(world).get("v2").approved_at == at(10)


def test_promote_unknown_request_and_bad_inputs_write_nothing(world):
    with pytest.raises(NotFoundError):
        promote(world, "APR-999999", actor="admin", role="admin", now=at(10))
    for kwargs in ({"actor": " "}, {"role": "boss"}, {"now": at(10).replace(tzinfo=None)}):
        with pytest.raises(ValueError):
            promote(world, "APR-000001", **{"actor": "admin", "role": "admin", "now": at(10), **kwargs})
    assert stage_of(world, "v1") is Stage.PRODUCTION and stage_of(world, "v2") is Stage.STAGING
    assert world.scalars(select(DeploymentRow)).first() is None and world.scalars(select(AuditLogRow)).first() is None


def test_promote_fails_if_the_candidate_is_no_longer_staging(world):
    world.get(ModelVersionRow, "v2").stage = "archived"
    world.flush()
    with pytest.raises(InvalidTransitionError):
        promote(world, "APR-000001", actor="admin", role="admin", now=at(10))
    assert ApprovalRepository(world).get("APR-000001").status is ApprovalStatus.PENDING


def test_failure_in_the_middle_undoes_everything(factory, monkeypatch):
    with session_scope(factory) as s:
        models = ModelVersionRepository(s)
        models.register(make_metrics("v1"), stage=Stage.PRODUCTION)
        models.register(GOOD, stage=Stage.STAGING)
        ApprovalRepository(s).add_pending("v2", requested_by="system", created_at=at(1))

    def broken(*args, **kwargs):
        raise RuntimeError("배포 기록 실패")

    monkeypatch.setattr(DeploymentRepository, "add", broken)
    with pytest.raises(RuntimeError):
        with session_scope(factory) as s:
            promote(s, "APR-000001", actor="admin", role="admin", now=at(10))
    with session_scope(factory) as s:  # 새 세션에서 본 DB 는 승인 전과 같다
        models = ModelVersionRepository(s)
        assert (models.get("v1").stage, models.get("v2").stage) == (Stage.PRODUCTION, Stage.STAGING)
        assert (models.get("v2").approved_by, models.get("v1").approved_by) == (None, None)
        assert ApprovalRepository(s).get("APR-000001").status is ApprovalStatus.PENDING
        assert s.scalars(select(AuditLogRow)).first() is None


# ---------- 거절 ----------

def test_reject_marks_request_and_candidate_and_keeps_production(world):
    result = reject(world, "APR-000001", actor="admin", role="admin", reason="수치가 애매함", now=at(10))
    assert (result.version, result.stage) == ("v2", Stage.REJECTED)
    models = ModelVersionRepository(world)
    assert (models.get("v1").stage, models.get("v2").stage) == (Stage.PRODUCTION, Stage.REJECTED)
    approval = ApprovalRepository(world).get("APR-000001")
    assert (approval.status, approval.reason, approval.handled_by, approval.handled_at) == (ApprovalStatus.REJECTED, "수치가 애매함", "admin", at(10))
    assert DeploymentRepository(world).list() == [] and AuditLogRepository(world).list()[0].action is AuditAction.REJECT


@pytest.mark.parametrize("reason", [None, "", "   "])
def test_reject_without_a_reason_is_refused_and_changes_nothing(world, reason):  # e2e X5
    with pytest.raises(ValueError):
        reject(world, "APR-000001", actor="admin", role="admin", reason=reason, now=at(10))
    assert stage_of(world, "v2") is Stage.STAGING and ApprovalRepository(world).get("APR-000001").status is ApprovalStatus.PENDING


# ---------- 롤백 ----------

@pytest.fixture
def after_promotion(world):
    """v2 가 승인되어 production, v1 은 archived (처음 승인 기록 APR-000001 은 approved)."""
    promote(world, "APR-000001", actor="admin", role="admin", now=at(10), reason="수치 확인")
    return world


def test_rollback_restores_the_archived_version_in_the_same_order(after_promotion):
    s = after_promotion
    statements = []

    @event.listens_for(s.get_bind(), "before_cursor_execute")
    def record(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("UPDATE model_version"):
            statements.append(parameters[0])

    try:
        result = rollback(s, "v1", actor="admin2", role="admin", reason="드리프트 오판 증가", now=at(20))
    finally:
        event.remove(s.get_bind(), "before_cursor_execute", record)
    assert statements == ["archived", "production"]
    assert (result.new_version, result.old_version, result.approval_id) == ("v1", "v2", None)
    models = ModelVersionRepository(s)
    assert (models.get("v1").stage, models.get("v2").stage) == (Stage.PRODUCTION, Stage.ARCHIVED)
    newest = DeploymentRepository(s).list()[0]
    assert (newest.version, newest.action.value, newest.reason, newest.actor, newest.at) == ("v1", "rollback", "드리프트 오판 증가", "admin2", at(20))
    log = AuditLogRepository(s).list()[0]
    assert (log.action, log.actor, log.target) == (AuditAction.ROLLBACK, "admin2", "v1")


def test_rollback_makes_approved_by_the_last_person_who_put_it_in_production(after_promotion):
    s = after_promotion
    assert ModelVersionRepository(s).get("v1").approved_by is None  # sync 로 처음 올린 v1 은 비어 있다
    rollback(s, "v1", actor="admin2", role="admin", reason="복귀", now=at(20))
    v1 = ModelVersionRepository(s).get("v1")
    assert (v1.approved_by, v1.approved_at) == ("admin2", at(20))
    rollback(s, "v2", actor="admin3", role="admin", reason="다시 v2", now=at(30))  # v2 도 다시 올릴 수 있다
    v2 = ModelVersionRepository(s).get("v2")
    assert (v2.approved_by, v2.approved_at) == ("admin3", at(30)) and ModelVersionRepository(s).get_production().version == "v2"


def test_rollback_leaves_the_original_approval_request_untouched(after_promotion):
    s = after_promotion
    before = ApprovalRepository(s).get("APR-000001")
    rollback(s, "v1", actor="admin2", role="admin", reason="복귀", now=at(20))
    rollback(s, "v2", actor="admin3", role="admin", reason="다시 v2", now=at(30))
    after = ApprovalRepository(s).get("APR-000001")
    assert after == before and (after.status, after.handled_by, after.handled_at, after.reason) == (ApprovalStatus.APPROVED, "admin", at(10), "수치 확인")
    assert ApprovalRepository(s).count() == 1  # 롤백은 승인 요청을 새로 만들지도 않는다


@pytest.mark.parametrize("reason", [None, "", "  "])
def test_rollback_without_a_reason_is_refused_and_changes_nothing(after_promotion, reason):  # e2e X5
    with pytest.raises(ValueError):
        rollback(after_promotion, "v1", actor="admin", role="admin", reason=reason, now=at(20))
    assert stage_of(after_promotion, "v1") is Stage.ARCHIVED and stage_of(after_promotion, "v2") is Stage.PRODUCTION
    assert len(DeploymentRepository(after_promotion).list()) == 1


def test_rollback_only_works_for_archived_versions(world):
    for version in ("v1", "v2", "v3"):  # production, staging, candidate
        with pytest.raises(InvalidTransitionError):
            rollback(world, version, actor="admin", role="admin", reason="시도", now=at(20))
    with pytest.raises(NotFoundError):
        rollback(world, "v9", actor="admin", role="admin", reason="시도", now=at(20))
    world.get(ModelVersionRow, "v3").stage = "rejected"
    world.flush()
    with pytest.raises(InvalidTransitionError):  # rejected 는 되살릴 수 없다
        rollback(world, "v3", actor="admin", role="admin", reason="시도", now=at(20))
    assert DeploymentRepository(world).list() == []


def test_rollback_with_no_current_production_just_restores_the_target(session):
    models = ModelVersionRepository(session)
    models.register(make_metrics("v1"), stage=Stage.ARCHIVED)
    result = rollback(session, "v1", actor="admin", role="admin", reason="운영 모델이 비어 있어서", now=at(1))
    assert result.old_version is None and models.get_production().version == "v1"


# ---------- 순서와 단독 감사 기록 ----------

def test_raising_a_candidate_before_archiving_the_old_production_is_refused(world):
    models = ModelVersionRepository(world)
    with pytest.raises(ConflictError):
        models.transition("v2", Stage.PRODUCTION, approved_by="admin", approved_at=at(10))
    assert stage_of(world, "v1") is Stage.PRODUCTION


def test_record_audit_can_be_called_alone_with_the_contract_actions(session):
    log = record_audit(session, actor="admin", role="admin", action="approve", target="v2", now=at(1))
    assert log.log_id == "LOG-000001" and log.action is AuditAction.APPROVE
    with pytest.raises(ValueError):
        record_audit(session, actor="admin", role="admin", action="reload_fail", target="v2", now=at(2))
    assert session.scalars(select(ApprovalRequestRow)).first() is None
