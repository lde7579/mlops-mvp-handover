"""파트 3 보조 테이블 저장소: approval_request, training_run, deployment, audit_log (계약 6~7장, C7)와 전이 메서드."""
from datetime import timedelta, timezone

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from backend.models import ModelVersionRow
from backend.repository import (
    ALLOWED_TRANSITIONS,
    ApprovalRepository,
    ApprovalStatus,
    AuditAction,
    AuditLogRepository,
    ConflictError,
    DeploymentAction,
    DeploymentRepository,
    InvalidTransitionError,
    ModelVersionRepository,
    NotFoundError,
    Role,
    RunStatus,
    RunStep,
    RunTrigger,
    Stage,
    TrainingRunRepository,
)
from tests.backend.helpers import KST, at, make_metrics


@pytest.fixture
def versions(session):
    repo = ModelVersionRepository(session)
    for n, stage in ((1, Stage.PRODUCTION), (2, Stage.STAGING), (3, Stage.CANDIDATE)):
        repo.register(make_metrics(f"v{n}", created_minutes=n), stage=stage)
    session.commit()  # 시험 안에서 rollback 해도 이 세 버전은 남는다
    return repo


# ---------- approval_request ----------

def test_add_pending_fills_contract_fields_and_issues_apr_ids(session, versions):
    repo = ApprovalRepository(session)
    first = repo.add_pending("v2", requested_by="system", created_at=at(1))
    second = repo.add_pending("v3", requested_by="system", created_at=at(2))
    assert (first.approval_id, second.approval_id) == ("APR-000001", "APR-000002")
    assert first.status is ApprovalStatus.PENDING and first.model_version == "v2" and first.requested_by == "system"
    assert (first.reason, first.handled_by, first.handled_at) == (None, None, None)
    assert first.created_at == at(1) and first.created_at.utcoffset() == timedelta(0)  # UTC 로 읽힌다
    assert repo.get("APR-000001") == first and repo.get("APR-999999") is None


def test_second_pending_for_the_same_version_is_a_conflict_but_after_handling_it_is_allowed(session, versions):
    repo = ApprovalRepository(session)
    first = repo.add_pending("v2", requested_by="system", created_at=at(1))
    with pytest.raises(ConflictError):
        repo.add_pending("v2", requested_by="system", created_at=at(2))
    assert repo.count() == 1
    repo.mark_handled(first.approval_id, status=ApprovalStatus.REJECTED, handled_by="admin", handled_at=at(3), reason="수치 부족")
    assert repo.add_pending("v2", requested_by="system", created_at=at(4)).approval_id == "APR-000002"


def test_database_itself_refuses_two_pending_for_one_version(session, versions):
    session.execute(text("INSERT INTO approval_request VALUES ('A1', 'v2', 's', 'pending', NULL, NULL, NULL, '2026-10-18T00:00:00.000000+00:00')"))
    with pytest.raises(IntegrityError):
        session.execute(text("INSERT INTO approval_request VALUES ('A2', 'v2', 's', 'pending', NULL, NULL, NULL, '2026-10-18T00:00:00.000000+00:00')"))
    session.rollback()


def test_add_pending_for_unknown_version_is_not_found(session, versions):
    with pytest.raises(NotFoundError):
        ApprovalRepository(session).add_pending("v9", requested_by="system", created_at=at())


def test_mark_handled_approve_and_reject_rules(session, versions):
    repo = ApprovalRepository(session)
    a = repo.add_pending("v2", requested_by="system", created_at=at(1))
    with pytest.raises(ValueError):  # 거절은 사유 필수
        repo.mark_handled(a.approval_id, status=ApprovalStatus.REJECTED, handled_by="admin", handled_at=at(2), reason="  ")
    with pytest.raises(ValueError):
        repo.mark_handled(a.approval_id, status=ApprovalStatus.PENDING, handled_by="admin", handled_at=at(2))
    assert repo.get(a.approval_id).status is ApprovalStatus.PENDING  # 실패한 시도는 아무것도 바꾸지 않았다
    done = repo.mark_handled(a.approval_id, status=ApprovalStatus.APPROVED, handled_by="admin", handled_at=at(2))
    assert done.status is ApprovalStatus.APPROVED and done.handled_by == "admin" and done.handled_at == at(2) and done.reason is None
    with pytest.raises(InvalidTransitionError):  # 이미 처리됨
        repo.mark_handled(a.approval_id, status=ApprovalStatus.APPROVED, handled_by="admin", handled_at=at(3))
    with pytest.raises(NotFoundError):
        repo.mark_handled("APR-999999", status=ApprovalStatus.APPROVED, handled_by="admin", handled_at=at(3))


def test_approval_list_is_newest_first_and_filters_by_status(session, versions):
    repo = ApprovalRepository(session)
    a = repo.add_pending("v2", requested_by="system", created_at=at(1))
    b = repo.add_pending("v3", requested_by="system", created_at=at(2))
    repo.mark_handled(a.approval_id, status=ApprovalStatus.APPROVED, handled_by="admin", handled_at=at(3))
    assert [x.approval_id for x in repo.list()] == [b.approval_id, a.approval_id]
    assert [x.approval_id for x in repo.list(status="pending")] == [b.approval_id] and repo.count(status="pending") == 1
    with pytest.raises(ValueError):
        repo.list(status="done")


# ---------- training_run ----------

def test_training_run_add_update_and_gate_result_round_trip(session, versions):
    repo = TrainingRunRepository(session)
    run = repo.add("manual")
    assert run.run_id == "RUN-000001" and run.status is RunStatus.QUEUED and run.trigger is RunTrigger.MANUAL
    assert (run.started_at, run.finished_at, run.result_version, run.step, run.error_message, run.gate_result) == (None,) * 6
    gate = {"baseline_version": "v1", "passed": False, "checks": [{"id": "G1", "passed": False}]}
    done = repo.update(run.run_id, status="succeeded", step="gate", started_at=at(1), finished_at=at(2), result_version="v3", gate_result=gate)
    assert done.status is RunStatus.SUCCEEDED and done.step is RunStep.GATE and done.result_version == "v3"
    assert done.gate_result == gate and done.started_at == at(1) and done.finished_at.tzinfo == timezone.utc
    assert repo.get("RUN-000001") == done and repo.add("count").run_id == "RUN-000002"
    assert [r.run_id for r in repo.list()] == ["RUN-000002", "RUN-000001"] and [r.run_id for r in repo.list(status="queued")] == ["RUN-000002"]


def test_training_run_update_rejects_bad_input(session, versions):
    repo = TrainingRunRepository(session)
    run = repo.add("performance")
    with pytest.raises(ValueError):
        repo.add("sometimes")
    for bad in ({"status": "done"}, {"step": "deploy"}, {"unknown": 1}, {"started_at": at().replace(tzinfo=None)}):
        with pytest.raises(ValueError):
            repo.update(run.run_id, **bad)
    with pytest.raises(NotFoundError):
        repo.update(run.run_id, result_version="v9")
    with pytest.raises(NotFoundError):
        repo.update("RUN-999999", status="running")
    failed = repo.update(run.run_id, status="failed", error_message="학습 중 오류")
    assert failed.status is RunStatus.FAILED and failed.error_message == "학습 중 오류" and failed.result_version is None


# ---------- deployment, audit_log ----------

def test_deployment_records_and_rollback_needs_a_reason(session, versions):
    repo = DeploymentRepository(session)
    deploy = repo.add("v2", "deploy", actor="admin", at=at(1))
    rollback = repo.add("v1", "rollback", actor="admin", at=at(2), reason="오판 증가")
    assert (deploy.deployment_id, rollback.deployment_id) == ("DEP-000001", "DEP-000002")
    assert deploy.action is DeploymentAction.DEPLOY and deploy.reason is None and rollback.reason == "오판 증가"
    assert [d.deployment_id for d in repo.list()] == ["DEP-000002", "DEP-000001"]
    for bad in (None, "", "  "):
        with pytest.raises(ValueError):
            repo.add("v1", "rollback", actor="admin", at=at(3), reason=bad)
    with pytest.raises(NotFoundError):
        repo.add("v9", "deploy", actor="admin", at=at(3))
    with pytest.raises(ValueError):
        repo.add("v1", "restart", actor="admin", at=at(3))
    assert len(repo.list()) == 2


def test_audit_log_accepts_only_contract_actions_and_roles(session):
    repo = AuditLogRepository(session)
    first = repo.add(actor="system", role="system", action="gate_pass", target="v2", at=at(1))
    repo.add(actor="admin", role=Role.ADMIN, action=AuditAction.APPROVE, target="v2", at=at(2))
    assert first.log_id == "LOG-000001" and first.role is Role.SYSTEM and first.action is AuditAction.GATE_PASS
    assert [x.log_id for x in repo.list()] == ["LOG-000002", "LOG-000001"]
    assert [x.log_id for x in repo.list(action="gate_pass")] == ["LOG-000001"]
    with pytest.raises(ValueError):
        repo.add(actor="admin", role="admin", action="reload_fail", target="v2", at=at(3))  # 이름은 4-C 에서 추가
    with pytest.raises(ValueError):
        repo.add(actor="admin", role="boss", action="approve", target="v2", at=at(3))
    with pytest.raises(ValueError):
        repo.add(actor="admin", role="admin", action="approve", target="v2", at=at(3).replace(tzinfo=None))


def test_times_are_stored_in_utc_and_ids_keep_counting(session, versions):
    repo = DeploymentRepository(session)
    repo.add("v1", "deploy", actor="admin", at=at(0).astimezone(KST))
    stored = session.execute(text("SELECT at FROM deployment")).scalar()
    assert stored == "2026-10-18T00:00:00.000000+00:00"
    for _ in range(11):
        last = repo.add("v1", "deploy", actor="admin", at=at(1))
    assert last.deployment_id == "DEP-000012"


# ---------- model_version 전이 ----------

ALL_STAGES = list(Stage)


def _version_in(session, stage, name="v7"):
    ModelVersionRepository(session).register(make_metrics(name), stage=Stage.CANDIDATE)
    session.get(ModelVersionRow, name).stage = stage.value
    session.flush()
    return name


@pytest.mark.parametrize("current,to", [(c, t) for c in ALL_STAGES for t in ALL_STAGES])
def test_transition_follows_the_contract_table(session, current, to):
    name = _version_in(session, current)
    repo = ModelVersionRepository(session)
    if (current, to) in ALLOWED_TRANSITIONS:
        done = repo.transition(name, to, approved_by="admin", approved_at=at(1))
        assert done.stage is to
        assert (done.approved_by, done.approved_at) == (("admin", at(1)) if to is Stage.PRODUCTION else (None, None))
    else:  # production -> archived 도 여기서는 막힌다 (archive_production 으로만)
        with pytest.raises(InvalidTransitionError):
            repo.transition(name, to, approved_by="admin", approved_at=at(1))
        assert repo.get(name).stage is current


def test_the_contract_table_has_exactly_five_direct_transitions_plus_archive_production():
    assert len(ALLOWED_TRANSITIONS) == 5


def test_archive_production_moves_only_the_production_and_none_is_fine(session, versions):
    repo = ModelVersionRepository(session)
    assert repo.archive_production().version == "v1" and repo.get("v1").stage is Stage.ARCHIVED
    assert repo.get_production() is None and repo.archive_production() is None


def test_production_needs_approver_and_a_free_slot(session, versions):
    repo = ModelVersionRepository(session)
    for bad in (None, "", " "):
        with pytest.raises(ValueError):
            repo.transition("v2", Stage.PRODUCTION, approved_by=bad, approved_at=at(1))
    with pytest.raises(ValueError):
        repo.transition("v2", Stage.PRODUCTION, approved_by="admin", approved_at=None)
    with pytest.raises(ConflictError):  # v1 이 아직 production
        repo.transition("v2", Stage.PRODUCTION, approved_by="admin", approved_at=at(1))
    assert repo.get("v2").stage is Stage.STAGING
    with pytest.raises(NotFoundError):
        repo.transition("v9", Stage.STAGING)


def test_database_refuses_a_second_production_when_the_old_one_is_not_archived_first(session, versions):
    session.get(ModelVersionRow, "v2").stage = "production"
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()
    assert [r for r in session.scalars(select(ModelVersionRow.version).where(ModelVersionRow.stage == "production"))] == ["v1"]
