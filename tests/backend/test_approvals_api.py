"""GET /approvals, GET /deployments, GET /audit-logs (api-v1.md A17, A21, A22). 읽기 전용 환경 하나를 모듈 안에서 함께 쓴다."""
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.database import session_scope
from backend.main import create_app
from backend.promotion import promote, record_audit, reject, rollback
from backend.repository import (
    ApprovalRepository,
    AuditAction,
    AuditLogRepository,
    DeploymentRepository,
    ModelVersionRepository,
    Role,
    Stage,
)
from tests.backend.helpers import at, bearer, make_metrics, make_settings

APPROVAL_KEYS = {"approval_id", "model_version", "requested_by", "status", "reason", "handled_by", "handled_at", "created_at"}
DEPLOYMENT_KEYS = {"deployment_id", "version", "action", "reason", "actor", "at"}
AUDIT_KEYS = {"log_id", "actor", "role", "action", "target", "at"}


def seed(session):
    """승인 요청 4건(승인 2·거절 1·대기 1), 배포 이력 3건, 감사 기록 6건."""
    models, approvals = ModelVersionRepository(session), ApprovalRepository(session)
    models.register(make_metrics("v1"), stage=Stage.PRODUCTION)
    for n, version in enumerate(("v2", "v3", "v4", "v5"), start=1):
        models.register(make_metrics(version, created_minutes=n), stage=Stage.STAGING)
    for n, version in enumerate(("v2", "v3", "v4", "v5"), start=1):
        approvals.add_pending(version, requested_by="system", created_at=at(n))
    record_audit(session, actor="system", role=Role.SYSTEM, action=AuditAction.GATE_PASS, target="v2", now=at(1))
    promote(session, "APR-000001", actor="admin", role="admin", now=at(10), reason="수치 확인")      # DEP-1 (사유 있음)
    reject(session, "APR-000002", actor="admin", role="admin", reason="수치 부족", now=at(11))
    promote(session, "APR-000003", actor="admin", role="admin", now=at(12))                         # DEP-2 (사유 없음)
    rollback(session, "v1", actor="admin", role="admin", reason="성능 확인", now=at(13))            # DEP-3
    record_audit(session, actor="admin", role=Role.ADMIN, action=AuditAction.APPROVE_FAIL, target="v5", now=at(14))


@pytest.fixture(scope="module")
def view():
    settings = make_settings()
    app = create_app(settings)
    with session_scope(app.state.session_factory) as s:
        seed(s)

    def counts():
        with session_scope(app.state.session_factory) as s:
            return (ApprovalRepository(s).count(), DeploymentRepository(s).count(), AuditLogRepository(s).count())

    assert counts() == (4, 3, 6)
    yield SimpleNamespace(client=TestClient(app, raise_server_exceptions=False),
                          op=bearer(settings, "operator"), admin=bearer(settings, "admin"))
    try:
        assert counts() == (4, 3, 6), "읽기 전용 환경의 행 수가 바뀌었습니다"
    finally:
        app.state.engine.dispose()


def get(view, path, who="op", **params):
    return view.client.get(path, params=params, headers=getattr(view, who))


def is_utc_z(text):
    return text.endswith("Z") and datetime.fromisoformat(text).utcoffset().total_seconds() == 0


# ---------- 권한 ----------

@pytest.mark.parametrize("path", ["/approvals", "/deployments", "/audit-logs"])
def test_login_is_required(view, path):
    r = view.client.get(path)
    assert r.status_code == 401 and r.json()["error"] == "unauthorized"


@pytest.mark.parametrize("path", ["/approvals", "/deployments"])
def test_operator_and_admin_can_read_approvals_and_deployments(view, path):
    assert get(view, path, "op").status_code == 200 and get(view, path, "admin").status_code == 200


def test_audit_logs_are_admin_only(view):
    r = get(view, "/audit-logs", "op")
    assert r.status_code == 403 and r.json()["error"] == "forbidden"
    assert get(view, "/audit-logs", "admin").status_code == 200


# ---------- GET /approvals ----------

def test_approvals_shape_order_and_total(view):
    body = get(view, "/approvals").json()
    assert set(body) == {"items", "total"} and body["total"] == 4
    assert all(set(i) == APPROVAL_KEYS for i in body["items"])
    assert [i["approval_id"] for i in body["items"]] == ["APR-000004", "APR-000003", "APR-000002", "APR-000001"]  # 최신 먼저
    pending, approved = body["items"][0], body["items"][3]
    assert (pending["status"], pending["reason"], pending["handled_by"], pending["handled_at"]) == ("pending", None, None, None)
    assert (approved["status"], approved["reason"], approved["handled_by"]) == ("approved", "수치 확인", "admin")
    assert is_utc_z(approved["handled_at"]) and is_utc_z(approved["created_at"])


def test_approvals_status_filter(view):
    assert [i["approval_id"] for i in get(view, "/approvals", status="pending").json()["items"]] == ["APR-000004"]
    rejected = get(view, "/approvals", status="rejected").json()
    assert rejected["total"] == 1 and rejected["items"][0]["reason"] == "수치 부족"
    assert get(view, "/approvals", status="approved").json()["total"] == 2


def test_approvals_paging_keeps_total(view):
    body = get(view, "/approvals", limit=1, offset=1).json()
    assert body["total"] == 4 and [i["approval_id"] for i in body["items"]] == ["APR-000003"]
    assert get(view, "/approvals", offset=10).json() == {"items": [], "total": 4}


@pytest.mark.parametrize("path,params", [
    ("/approvals", {"status": "done"}), ("/approvals", {"limit": 0}), ("/approvals", {"limit": 1001}), ("/approvals", {"offset": -1}),
    ("/deployments", {"action": "undo"}), ("/deployments", {"limit": 0}), ("/deployments", {"offset": -1}),
    ("/audit-logs", {"action": "undo"}), ("/audit-logs", {"limit": 1001}), ("/audit-logs", {"offset": -1}),
])
def test_bad_query_values_are_400(view, path, params):
    r = get(view, path, "admin", **params)
    assert r.status_code == 400 and r.json()["error"] == "bad_request"


# ---------- GET /deployments ----------

def test_deployments_shape_and_order(view):
    body = get(view, "/deployments").json()
    assert body["total"] == 3 and all(set(i) == DEPLOYMENT_KEYS for i in body["items"])
    assert [(i["deployment_id"], i["action"], i["version"]) for i in body["items"]] == [
        ("DEP-000003", "rollback", "v1"), ("DEP-000002", "deploy", "v4"), ("DEP-000001", "deploy", "v2")]
    assert body["items"][0]["reason"] == "성능 확인" and body["items"][1]["reason"] is None
    assert body["items"][0]["actor"] == "admin" and is_utc_z(body["items"][0]["at"])


def test_deployments_filters_and_paging(view):
    assert [i["deployment_id"] for i in get(view, "/deployments", version="v1").json()["items"]] == ["DEP-000003"]
    assert get(view, "/deployments", action="deploy").json()["total"] == 2
    assert get(view, "/deployments", version="v1", action="deploy").json() == {"items": [], "total": 0}
    assert get(view, "/deployments", version="v99").json() == {"items": [], "total": 0}
    page = get(view, "/deployments", limit=1, offset=2).json()
    assert page["total"] == 3 and [i["deployment_id"] for i in page["items"]] == ["DEP-000001"]


# ---------- GET /audit-logs ----------

def test_audit_logs_shape_and_order(view):
    body = get(view, "/audit-logs", "admin").json()
    assert body["total"] == 6 and all(set(i) == AUDIT_KEYS for i in body["items"])
    assert [i["action"] for i in body["items"]] == ["approve_fail", "rollback", "approve", "reject", "approve", "gate_pass"]
    first, last = body["items"][0], body["items"][-1]
    assert (first["actor"], first["role"], first["target"]) == ("admin", "admin", "v5") and is_utc_z(first["at"])
    assert (last["actor"], last["role"]) == ("system", "system")


def test_audit_logs_filter_accepts_the_failure_actions_and_paging(view):
    assert get(view, "/audit-logs", "admin", action="approve_fail").json()["total"] == 1
    assert get(view, "/audit-logs", "admin", action="rollback_fail").json() == {"items": [], "total": 0}
    assert get(view, "/audit-logs", "admin", action="approve").json()["total"] == 2
    page = get(view, "/audit-logs", "admin", limit=2, offset=1).json()
    assert page["total"] == 6 and [i["action"] for i in page["items"]] == ["rollback", "approve"]
