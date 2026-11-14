"""POST /approvals/{id}/reject, /approvals/{id}/approve, /models/{version}/rollback (api-v1.md A18~A20, e2e X2·X4·X5·X6).
메모리 DB, 시험마다 새 앱. 가짜 모델 로더로 추론부를 바꿔 끼우므로 torch 를 가져오지 않는다."""
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from backend import deployment
from backend.database import session_scope
from backend.deployment import DeployService
from backend.inference import InferenceService
from backend.main import create_app
from backend.models import ApprovalRequestRow, AuditLogRow, DeploymentRow, ModelVersionRow
from backend.promotion import reject as promotion_reject
from backend.repository import (
    ApprovalRepository,
    InspectionRepository,
    ModelVersionRepository,
    Stage,
    TrainingRunRepository,
)
from tests.backend.helpers import FakeParts, at, bearer, make_inspection, make_metrics, make_settings, write_png


TABLES = ("prediction", "approval_request", "deployment", "audit_log", "training_run", "inspection", "model_version")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """앱과 클라이언트는 모듈당 하나만 만들고(앱을 만들고 첫 요청을 처리하는 데 시험마다 약 60ms), 시험마다 fresh() 로 DB 를 비우고 다시 채운다."""
    root = tmp_path_factory.mktemp("deploy")
    settings = make_settings(root=str(root))
    app = create_app(settings)
    write_png(root, "data/pool/INS-000001.png", np.zeros((128, 128), dtype=np.uint8))
    yield SimpleNamespace(app=app, settings=settings, client=TestClient(app, raise_server_exceptions=False),
                          op=bearer(settings, "operator"), admin=bearer(settings, "admin"))
    app.state.engine.dispose()


def fresh(world, staging=("v2",), production=True):
    """DB 를 비우고 다시 채운다: v1 production(production=False 면 없음), staging 후보마다 승인 요청 1건(APR-000001부터 순서대로),
    판정할 검사 1건. 추론부와 배포 서비스도 새로 만들어 가짜 모델 로더(FakeParts)를 쓴다."""
    app = world.app
    with session_scope(app.state.session_factory) as s:
        for table in TABLES:
            s.execute(text(f"DELETE FROM {table}"))
        models, approvals = ModelVersionRepository(s), ApprovalRepository(s)
        if production:
            models.register(make_metrics("v1"), stage=Stage.PRODUCTION)
        for n, version in enumerate(staging, start=1):
            models.register(make_metrics(version, created_minutes=n), stage=Stage.STAGING)
            approvals.add_pending(version, requested_by="system", created_at=at(n))
        InspectionRepository(s).add(make_inspection(1))
    fake = FakeParts()
    inference = InferenceService(world.settings, parts_loader=fake)
    app.state.inference = inference
    app.state.deploy = DeployService(app.state.session_factory, inference)
    world.fake, world.inference = fake, inference
    return world


def add_run(env, version, baseline):
    """version 을 결과로 가진 학습 기록과 게이트 결과(기준 모델 baseline)를 넣는다."""
    with session_scope(env.app.state.session_factory) as s:
        runs = TrainingRunRepository(s)
        run = runs.add("manual")
        runs.update(run.run_id, status="succeeded", result_version=version,
                    gate_result={"baseline_version": baseline, "passed": True, "checks": []})


def snapshot(env):
    """DB 의 모델 stage, 승인 요청 상태, 배포·감사 행 수. 요청이 아무것도 바꾸지 않았는지 비교하는 데 쓴다."""
    with session_scope(env.app.state.session_factory) as s:
        return (
            {r.version: r.stage for r in s.scalars(select(ModelVersionRow))},
            {r.approval_id: (r.status, r.reason, r.handled_by) for r in s.scalars(select(ApprovalRequestRow))},
            len(s.scalars(select(DeploymentRow)).all()),
            len(s.scalars(select(AuditLogRow)).all()),
        )


@pytest.fixture
def env(world):
    return fresh(world)


def audit_actions(env):
    with session_scope(env.app.state.session_factory) as s:
        return [(r.action, r.actor, r.role, r.target) for r in s.scalars(select(AuditLogRow).order_by(AuditLogRow.log_id))]


# ---------- 권한 (관리자 전용 첫 엔드포인트: 403 경로를 증명한다) ----------

WRITES = [("/approvals/APR-000001/reject", {"reason": "수치 부족"}), ("/approvals/APR-000001/approve", {"reason": "확인"}),
          ("/models/v1/rollback", {"reason": "성능 저하"})]


@pytest.mark.parametrize("path,body", WRITES)
def test_login_is_required(env, path, body):
    before = snapshot(env)
    r = env.client.post(path, json=body)
    assert r.status_code == 401 and r.json()["error"] == "unauthorized" and r.headers["www-authenticate"] == "Bearer"
    assert snapshot(env) == before


@pytest.mark.parametrize("path,body", WRITES)
def test_operator_gets_403_and_nothing_changes(env, path, body):
    before = snapshot(env)
    r = env.client.post(path, json=body, headers=env.op)
    assert r.status_code == 403 and r.json()["error"] == "forbidden"
    assert snapshot(env) == before


@pytest.mark.parametrize("path,body", [("/approvals/APR-999999/reject", {"reason": "x"}), ("/approvals/abc/reject", {}),
                                       ("/approvals/APR-999999/approve", {"reason": "x"}), ("/approvals/abc/approve", {"reason": "  "}),
                                       ("/models/v99/rollback", {"reason": "x"}), ("/models/abc/rollback", {})])
def test_403_comes_before_404_and_400(env, path, body):
    assert env.client.post(path, json=body, headers=env.op).status_code == 403


# ---------- reject ----------

def test_reject_succeeds_and_records_everything(env):
    r = env.client.post("/approvals/APR-000001/reject", json={"reason": "수치 부족"}, headers=env.admin)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"approval", "version", "stage"} and (body["version"], body["stage"]) == ("v2", "rejected")
    assert (body["approval"]["status"], body["approval"]["reason"], body["approval"]["handled_by"]) == ("rejected", "수치 부족", "admin")
    assert body["approval"]["handled_at"].endswith("Z")
    stages, approvals, deployments, audits = snapshot(env)
    assert stages == {"v1": "production", "v2": "rejected"} and deployments == 0 and audits == 1
    assert audit_actions(env) == [("reject", "admin", "admin", "v2")]
    assert env.app.state.inference.loaded_version is None  # 거절은 추론부 모델을 건드리지 않는다


@pytest.mark.parametrize("body", [None, {}, {"reason": ""}, {"reason": "   "}, {"reason": None}, {"reason": "x", "extra": 1}, {"reason": 5}])
def test_reject_without_a_usable_reason_is_400(env, body):
    before = snapshot(env)
    r = env.client.post("/approvals/APR-000001/reject", json=body, headers=env.admin)
    assert r.status_code == 400 and r.json()["error"] == "bad_request"
    assert snapshot(env) == before


def test_reject_of_unknown_or_malformed_id_is_404(env):
    for approval_id in ("APR-999999", "abc", "APR-1x", "apr-000001"):
        r = env.client.post(f"/approvals/{approval_id}/reject", json={"reason": "x"}, headers=env.admin)
        assert r.status_code == 404 and r.json()["error"] == "not_found", approval_id


def test_400_comes_before_404(env):
    assert env.client.post("/approvals/APR-999999/reject", json={}, headers=env.admin).status_code == 400


def test_reject_twice_is_409_and_keeps_the_first_record(env):
    assert env.client.post("/approvals/APR-000001/reject", json={"reason": "첫 번째"}, headers=env.admin).status_code == 200
    before = snapshot(env)
    r = env.client.post("/approvals/APR-000001/reject", json={"reason": "두 번째"}, headers=env.admin)
    assert r.status_code == 409 and r.json()["error"] == "conflict"
    assert snapshot(env) == before and before[1]["APR-000001"] == ("rejected", "첫 번째", "admin")


# ---------- approve ----------

def predict(env):
    return env.client.post("/predict", json={"inspection_id": "INS-000001"}, headers=env.op)


def approve(env, approval_id="APR-000001", json=None, headers=None):
    return env.client.post(f"/approvals/{approval_id}/approve", json=json, headers=env.admin if headers is None else headers)


def test_approve_promotes_swaps_the_model_and_answers_after_both_are_done(env):
    assert predict(env).json()["model_version"] == "v1"  # 승인 전 판정은 v1
    r = approve(env, json={"reason": "수치 확인"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"approval", "deployment", "new_version", "old_version", "reload"}
    assert (body["new_version"], body["old_version"]) == ("v2", "v1")
    assert (body["approval"]["status"], body["approval"]["reason"], body["approval"]["handled_by"]) == ("approved", "수치 확인", "admin")
    assert (body["deployment"]["deployment_id"], body["deployment"]["action"], body["deployment"]["version"],
            body["deployment"]["actor"], body["deployment"]["reason"]) == ("DEP-000001", "deploy", "v2", "admin", "수치 확인")
    assert body["reload"]["loaded_version"] == "v2" and isinstance(body["reload"]["elapsed_ms"], int) and body["reload"]["elapsed_ms"] >= 0
    # 응답을 받은 시점에 이미 /health 와 /predict 가 새 버전을 쓴다
    assert env.client.get("/health").json()["model"] == {"production_version": "v2", "loaded": True}
    after = predict(env).json()
    assert (after["model_version"], after["threshold"]) == ("v2", 0.7)
    stages, approvals, deployments, audits = snapshot(env)
    assert stages == {"v1": "archived", "v2": "production"} and deployments == 1 and audits == 1
    assert audit_actions(env) == [("approve", "admin", "admin", "v2")]
    model = env.client.get("/models/v2", headers=env.op).json()
    assert model["approved_by"] == "admin" and model["approved_at"].endswith("Z")
    assert env.fake.loads == ["v1", "v2"]  # v2 파일은 읽어 두기 때 한 번만 읽었다


@pytest.mark.parametrize("body", [None, {}, {"reason": None}])
def test_approve_reason_is_optional(env, body):
    r = approve(env, json=body)
    assert r.status_code == 200 and r.json()["approval"]["reason"] is None and r.json()["deployment"]["reason"] is None


@pytest.mark.parametrize("body", [{"reason": ""}, {"reason": "   "}, {"reason": 5}, {"reason": "x", "extra": 1}, {"other": 1}])
def test_approve_with_a_bad_body_is_400_and_reads_no_model(env, body):
    before = snapshot(env)
    r = approve(env, json=body)
    assert r.status_code == 400 and r.json()["error"] == "bad_request"
    assert snapshot(env) == before and env.fake.loads == []


def test_approve_unknown_or_malformed_id_is_404_without_reading_a_model(env):
    for approval_id in ("APR-999999", "abc", "APR-1x"):
        r = approve(env, approval_id)
        assert r.status_code == 404 and r.json()["error"] == "not_found", approval_id
    assert env.fake.loads == []


def test_approving_an_already_handled_request_is_409_without_reading_a_model(env):
    assert approve(env).status_code == 200
    env.fake.loads.clear()
    before = snapshot(env)
    r = approve(env)
    assert r.status_code == 409 and r.json()["error"] == "conflict"
    assert snapshot(env) == before and env.fake.loads == []


def test_approving_a_rejected_request_is_409(env):
    assert env.client.post("/approvals/APR-000001/reject", json={"reason": "수치 부족"}, headers=env.admin).status_code == 200
    r = approve(env)
    assert r.status_code == 409 and env.fake.loads == []


def test_error_order_is_401_403_400_404_409_503(world):
    env = fresh(world, staging=("v2", "v3"))
    bad_body, unknown = {"reason": "   "}, "APR-999999"
    assert env.client.post(f"/approvals/{unknown}/approve", json=bad_body).status_code == 401
    assert approve(env, unknown, bad_body, env.op).status_code == 403
    assert approve(env, unknown, bad_body).status_code == 400
    assert approve(env, unknown, {"reason": "ok"}).status_code == 404
    env.fake.broken.update({"v2", "v3"})  # 새 모델 파일이 깨져 있어도, 먼저 걸리는 오류가 우선한다
    assert env.client.post("/approvals/APR-000001/reject", json={"reason": "수치 부족"}, headers=env.admin).status_code == 200
    assert approve(env).status_code == 409 and env.fake.loads == []
    assert approve(env, "APR-000002").status_code == 503  # 처리 대기 요청 + 깨진 모델 파일(v3 도 깨져 있다)


# ---------- approve: 새 모델을 불러오지 못할 때 (e2e X2) ----------

def test_load_failure_is_503_keeps_the_old_model_and_leaves_only_an_audit_record(env):
    assert predict(env).json()["model_version"] == "v1"
    before = snapshot(env)
    env.fake.broken.add("v2")
    r = approve(env)
    assert r.status_code == 503 and r.json()["error"] == "service_unavailable"
    assert "v2" in r.json()["message"] and "기존 운영 모델 v1" in r.json()["message"]
    stages, approvals, deployments, audits = snapshot(env)
    assert (stages, approvals, deployments) == before[:3] and audits == before[3] + 1
    assert approvals["APR-000001"][0] == "pending"  # 승인 요청은 대기로 남는다
    assert audit_actions(env)[-1] == ("approve_fail", "admin", "admin", "v2")  # 별도 세션이라 되돌려지지 않았다
    assert env.inference.loaded_version == "v1" and predict(env).json()["model_version"] == "v1"
    assert env.client.get("/health").json()["model"] == {"production_version": "v1", "loaded": True}
    env.fake.broken.clear()  # 파일을 고치면 같은 요청을 다시 승인할 수 있다
    assert approve(env).status_code == 200 and predict(env).json()["model_version"] == "v2"


def test_load_failure_without_any_production_model_says_so(world):
    env = fresh(world, production=False)
    env.fake.broken.add("v2")
    r = approve(env)
    assert r.status_code == 503 and "운영 모델은 바뀌지 않았습니다" in r.json()["message"]


# ---------- approve: 기준 모델 검사 (api-v1.md 4장) ----------

def test_a_second_candidate_judged_against_an_old_baseline_is_409(world):
    env = fresh(world, staging=("v2", "v3"))
    add_run(env, "v2", "v1")
    add_run(env, "v3", "v1")  # v3 도 v1 을 기준으로 게이트를 통과했다
    assert approve(env, "APR-000001").status_code == 200
    env.fake.loads.clear()
    before = snapshot(env)
    r = approve(env, "APR-000002")
    assert r.status_code == 409 and "기준 모델(v1)" in r.json()["message"] and "운영 모델(v2)" in r.json()["message"]
    assert snapshot(env) == before and env.fake.loads == []
    assert env.client.post("/approvals/APR-000002/reject", json={"reason": "기준 변경"}, headers=env.admin).status_code == 200


def test_baseline_check_passes_when_both_are_absent(world):
    env = fresh(world, production=False)  # 운영 모델이 없을 때 게이트를 통과한 첫 후보(baseline 없음)
    add_run(env, "v2", None)
    r = approve(env)
    assert r.status_code == 200 and r.json()["old_version"] is None


def test_baseline_none_but_a_production_exists_now_is_409(env):
    add_run(env, "v2", None)
    r = approve(env)
    assert r.status_code == 409 and "기준 모델(없음)" in r.json()["message"]


# ---------- approve: 읽어 둔 뒤 ~ 커밋 사이 ----------

def test_request_handled_by_someone_else_while_reading_is_rechecked_and_409(env):
    """읽어 두는 동안 다른 관리자가 거절했다: 쓰기 트랜잭션에서 다시 확인해 409, 읽어 둔 모델은 버린다."""
    def reject_meanwhile(version):
        with session_scope(env.app.state.session_factory) as s:
            promotion_reject(s, "APR-000001", actor="admin2", role="admin", reason="먼저 거절", now=at(20))

    env.fake.on_load = reject_meanwhile
    r = approve(env)
    assert r.status_code == 409
    stages, approvals, deployments, audits = snapshot(env)
    assert stages["v2"] == "rejected" and approvals["APR-000001"] == ("rejected", "먼저 거절", "admin2") and deployments == 0
    assert env.inference._prepared is None and env.inference.loaded_version is None


def test_unexpected_failure_while_writing_is_500_and_throws_the_prepared_model_away(env, monkeypatch):
    assert predict(env).json()["model_version"] == "v1"
    before = snapshot(env)

    def boom(*args, **kwargs):
        raise RuntimeError("디스크 오류")

    monkeypatch.setattr(deployment, "promote", boom)
    r = approve(env)
    assert r.status_code == 500 and r.json()["error"] == "server_error"
    assert snapshot(env) == before and env.inference._prepared is None
    assert env.inference.loaded_version == "v1" and predict(env).json()["model_version"] == "v1"


def test_commit_done_but_swap_not_run_recovers_on_the_next_predict(env, monkeypatch):
    """커밋은 됐는데 교체가 실행되지 못한 상황: 응답은 막지 않고, 다음 /predict 가 DB 의 production 으로 v2 를 불러온다."""
    assert predict(env).json()["model_version"] == "v1"

    def broken_swap(prepared):
        raise RuntimeError("교체 실패")

    monkeypatch.setattr(env.inference, "activate", broken_swap)
    r = approve(env)
    assert r.status_code == 200 and r.json()["new_version"] == "v2"
    assert env.client.get("/models/production", headers=env.op).json()["version"] == "v2"
    assert env.inference.loaded_version == "v1"
    assert env.client.get("/health").json()["model"] == {"production_version": "v2", "loaded": False}  # 아직 교체 전이라는 사실이 보인다
    after = predict(env)
    assert after.status_code == 201 and after.json()["model_version"] == "v2" and after.json()["threshold"] == 0.7
    assert env.inference.loaded_version == "v2" and env.fake.loads == ["v1", "v2"]  # 읽어 둔 모델을 쓴다 (다시 읽지 않았다)
    assert env.client.get("/health").json()["model"] == {"production_version": "v2", "loaded": True}


# ---------- rollback ----------

def rollback(env, version="v1", json=None, headers=None):
    return env.client.post(f"/models/{version}/rollback", json=json, headers=env.admin if headers is None else headers)


def approved_env(world, **kwargs):
    """v2 를 승인해 v1 이 archived, v2 가 production 인 상태 (추론부는 v2 를 올려 둔 상태)."""
    env = fresh(world, **kwargs)
    assert approve(env).status_code == 200
    env.fake.loads.clear()
    return env


def test_rollback_restores_the_old_version_and_answers_after_the_swap(world):
    env = approved_env(world)
    assert predict(env).json()["model_version"] == "v2"
    r = rollback(env, json={"reason": "성능 저하"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"deployment", "new_version", "old_version", "reload"} and (body["new_version"], body["old_version"]) == ("v1", "v2")
    assert (body["deployment"]["deployment_id"], body["deployment"]["action"], body["deployment"]["version"],
            body["deployment"]["actor"], body["deployment"]["reason"]) == ("DEP-000002", "rollback", "v1", "admin", "성능 저하")
    assert body["reload"]["loaded_version"] == "v1" and isinstance(body["reload"]["elapsed_ms"], int)
    assert env.client.get("/health").json()["model"] == {"production_version": "v1", "loaded": True}
    after = predict(env).json()
    assert (after["model_version"], after["threshold"]) == ("v1", 0.5)
    stages, approvals, deployments, audits = snapshot(env)
    assert stages == {"v1": "production", "v2": "archived"} and deployments == 2
    assert approvals["APR-000001"][0] == "approved"  # 승인 요청 원본은 롤백해도 바뀌지 않는다
    assert audit_actions(env)[-1] == ("rollback", "admin", "admin", "v1")
    assert env.client.get("/models/v1", headers=env.op).json()["approved_by"] == "admin"  # 마지막으로 production 에 올린 사람 (계약 C7)
    assert env.fake.loads == ["v1"]


@pytest.mark.parametrize("body", [None, {}, {"reason": ""}, {"reason": "   "}, {"reason": None}, {"reason": "x", "extra": 1}])
def test_rollback_without_a_usable_reason_is_400_and_reads_no_model(world, body):
    env = approved_env(world)
    before = snapshot(env)
    r = rollback(env, json=body)
    assert r.status_code == 400 and r.json()["error"] == "bad_request"
    assert snapshot(env) == before and env.fake.loads == []


def test_rollback_of_unknown_or_malformed_version_is_404(world):
    env = approved_env(world)
    for version in ("v99", "abc", "V1", "v1x"):
        r = rollback(env, version, {"reason": "x"})
        assert r.status_code == 404 and r.json()["error"] == "not_found", version
    assert env.fake.loads == []


def test_only_archived_versions_can_be_rolled_back(world):
    env = fresh(world, staging=("v2", "v3"))
    assert env.client.post("/approvals/APR-000002/reject", json={"reason": "수치 부족"}, headers=env.admin).status_code == 200
    for version, stage in (("v1", "production"), ("v2", "staging"), ("v3", "rejected")):
        before = snapshot(env)
        r = rollback(env, version, {"reason": "되돌리기"})
        assert r.status_code == 409 and r.json()["error"] == "conflict" and stage in r.json()["message"], version
        assert snapshot(env) == before
    assert env.fake.loads == []


def test_rollback_error_order_is_401_403_400_404_409_503(world):
    env = approved_env(world)
    assert env.client.post("/models/v99/rollback", json={"reason": "  "}).status_code == 401
    assert rollback(env, "v99", {"reason": "  "}, env.op).status_code == 403
    assert rollback(env, "v99", {"reason": "  "}).status_code == 400
    assert rollback(env, "v99", {"reason": "ok"}).status_code == 404
    env.fake.broken.update({"v1", "v2"})  # 모델 파일이 깨져 있어도, 먼저 걸리는 오류가 우선한다
    assert rollback(env, "v2", {"reason": "ok"}).status_code == 409  # v2 는 지금 production
    assert env.fake.loads == []
    assert rollback(env, "v1", {"reason": "ok"}).status_code == 503


# ---------- rollback: 모델 파일을 불러오지 못할 때 ----------

def test_rollback_load_failure_is_503_and_keeps_the_current_model(world):
    env = approved_env(world)
    assert predict(env).json()["model_version"] == "v2"
    before = snapshot(env)
    env.fake.broken.add("v1")
    r = rollback(env, json={"reason": "성능 저하"})
    assert r.status_code == 503 and r.json()["error"] == "service_unavailable"
    assert r.json()["message"] == "모델 v1 을 불러오지 못해 롤백하지 않았습니다. 기존 운영 모델 v2 을 계속 사용합니다"
    stages, approvals, deployments, audits = snapshot(env)
    assert (stages, approvals, deployments) == before[:3] and audits == before[3] + 1
    assert audit_actions(env)[-1] == ("rollback_fail", "admin", "admin", "v1")
    assert env.inference.loaded_version == "v2" and predict(env).json()["model_version"] == "v2"
    env.fake.broken.clear()  # 파일을 고치면 다시 롤백할 수 있다
    assert rollback(env, json={"reason": "성능 저하"}).status_code == 200 and predict(env).json()["model_version"] == "v1"


def test_unexpected_failure_while_writing_a_rollback_is_500_and_throws_the_prepared_model_away(world, monkeypatch):
    env = approved_env(world)
    before = snapshot(env)

    def boom(*args, **kwargs):
        raise RuntimeError("디스크 오류")

    monkeypatch.setattr(deployment, "promotion_rollback", boom)
    r = rollback(env, json={"reason": "성능 저하"})
    assert r.status_code == 500 and snapshot(env) == before and env.inference._prepared is None
    assert env.inference.loaded_version == "v2"
