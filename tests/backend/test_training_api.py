"""POST /training/runs, GET /training/runs, GET /training/runs/{id} (api-v1.md A15, A16). 가짜 학습기, 임시 폴더, 메모리 DB."""
import threading
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.database import session_scope
from backend.main import create_app
from backend.repository import InspectionRepository, ModelVersionRepository, Stage, TrainingRunRepository
from backend.retrain import BUSY, INTERRUPTED, NO_POOL, RetrainService
from tests.backend.helpers import bearer, make_inspection, make_metrics, make_settings
from tests.backend.test_retrain_service import FakeTrainer, fake_fixed_loader

RUN_KEYS = {"run_id", "status", "trigger", "started_at", "finished_at", "result_version", "step", "error_message",
            "gate_result", "train_data_snapshot"}


def build(tmp_path, *, production=True, dataset=True, pool=2, trainer=None):
    settings = make_settings(root=str(tmp_path))
    app = create_app(settings)
    if dataset:
        (tmp_path / "data" / "dataset").mkdir(parents=True)
        for name in ("manifest.jsonl", "dataset_info.json"):
            (tmp_path / "data" / "dataset" / name).write_text("", encoding="utf-8")
    with session_scope(app.state.session_factory) as s:
        if production:
            ModelVersionRepository(s).register(make_metrics("v1"), stage=Stage.PRODUCTION)
        for n in range(1, pool + 1):
            path = f"data/pool/INS-{n:06d}.png"
            (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((128, 128), 100 + n, dtype=np.uint8)).save(tmp_path / path)
            InspectionRepository(s).add(make_inspection(n, data_split="pool", image_path=path))
    trainer = trainer or FakeTrainer()
    app.state.retrain = RetrainService(settings, app.state.session_factory, trainer=trainer, fixed_loader=fake_fixed_loader)
    return SimpleNamespace(app=app, settings=settings, trainer=trainer, retrain=app.state.retrain,
                           client=TestClient(app, raise_server_exceptions=False),
                           op=bearer(settings, "operator"), admin=bearer(settings, "admin"))


@pytest.fixture
def env(tmp_path):
    e = build(tmp_path)
    yield e
    e.retrain.wait(20)
    e.app.state.engine.dispose()


def post(env, json=None, headers=None, **kw):
    return env.client.post("/training/runs", json=json, headers=env.op if headers is None else headers, **kw)


def finish(env):
    assert env.retrain.wait(20)


# ---------- 권한 ----------

@pytest.mark.parametrize("method,path", [("post", "/training/runs"), ("get", "/training/runs"), ("get", "/training/runs/RUN-000001")])
def test_login_is_required(env, method, path):
    r = getattr(env.client, method)(path)
    assert r.status_code == 401 and r.json()["error"] == "unauthorized"


@pytest.mark.parametrize("who", ["op", "admin"])
def test_operator_and_admin_can_start_and_read(env, who):
    headers = getattr(env, who)
    assert post(env, {"trigger": "manual"}, headers).status_code == 201
    finish(env)
    assert env.client.get("/training/runs", headers=headers).status_code == 200
    assert env.client.get("/training/runs/RUN-000001", headers=headers).status_code == 200


# ---------- POST ----------

def test_post_returns_201_with_a_queued_run_of_10_fields(env):
    r = post(env, {"trigger": "manual"})
    body = r.json()
    assert r.status_code == 201 and set(body) == RUN_KEYS
    assert (body["run_id"], body["status"], body["trigger"]) == ("RUN-000001", "queued", "manual")
    assert all(body[k] is None for k in ("started_at", "finished_at", "result_version", "step", "error_message", "gate_result", "train_data_snapshot"))


@pytest.mark.parametrize("kwargs", [{}, {"json": {}}, {"json": {"trigger": "manual"}}])
def test_body_is_optional_and_defaults_to_manual(tmp_path, kwargs):
    e = build(tmp_path)
    try:
        r = e.client.post("/training/runs", headers=e.op, **kwargs)
        assert r.status_code == 201 and r.json()["trigger"] == "manual"
    finally:
        e.retrain.wait(20)
        e.app.state.engine.dispose()


@pytest.mark.parametrize("payload", [{"trigger": "count"}, {"trigger": "performance"}, {"trigger": "auto"}, {"trigger": 1},
                                     {"trigger": None}, {"trigger": "manual", "extra": 1}, {"version": "v9"}])
def test_other_triggers_and_extra_fields_are_400_and_create_nothing(env, payload):
    r = post(env, payload)
    assert r.status_code == 400 and r.json()["error"] == "bad_request"
    with session_scope(env.app.state.session_factory) as s:
        assert TrainingRunRepository(s).count() == 0
    assert env.trainer.calls == []


def test_non_object_body_is_400(env):
    assert env.client.post("/training/runs", content="[]", headers={**env.op, "Content-Type": "application/json"}).status_code == 400
    assert env.client.post("/training/runs", content="not json", headers={**env.op, "Content-Type": "application/json"}).status_code == 400


def test_second_post_while_running_is_409_with_the_documented_message(tmp_path):
    release = threading.Event()
    e = build(tmp_path, trainer=FakeTrainer(gate=release))
    try:
        assert e.client.post("/training/runs", headers=e.op).status_code == 201  # 학습이 풀리기 전에 이미 응답이 왔다
        r = e.client.post("/training/runs", headers=e.op)
        assert r.status_code == 409 and r.json() == {"error": "conflict", "message": BUSY}
        mid = e.client.get("/training/runs", headers=e.op).json()
        assert mid["total"] == 1 and mid["items"][0]["status"] in ("queued", "running")
    finally:
        release.set()
        e.retrain.wait(20)
        e.app.state.engine.dispose()


def test_no_pool_data_is_409_and_no_dataset_is_503(tmp_path):
    e = build(tmp_path, pool=0)
    r = e.client.post("/training/runs", headers=e.op)
    assert r.status_code == 409 and r.json() == {"error": "conflict", "message": NO_POOL}
    e.app.state.engine.dispose()
    e = build(tmp_path / "other", dataset=False)
    r = e.client.post("/training/runs", headers=e.op)
    assert r.status_code == 503 and r.json()["error"] == "service_unavailable"
    with session_scope(e.app.state.session_factory) as s:
        assert TrainingRunRepository(s).count() == 0
    e.app.state.engine.dispose()


# ---------- GET ----------

def test_finished_run_has_gate_result_snapshot_and_utc_times(env):
    post(env)
    finish(env)
    body = env.client.get("/training/runs/RUN-000001", headers=env.op).json()
    assert set(body) == RUN_KEYS
    assert (body["status"], body["step"], body["result_version"], body["error_message"]) == ("succeeded", "gate", "v2", None)
    assert body["gate_result"]["passed"] is True and body["gate_result"]["baseline_version"] == "v1"
    assert body["train_data_snapshot"] == {"train": 4, "pool": 2, "seed": 42}
    for key in ("started_at", "finished_at"):
        assert body[key].endswith("Z") and datetime.fromisoformat(body[key]).utcoffset().total_seconds() == 0
    assert env.client.get("/models/v2", headers=env.op).json()["stage"] == "staging"  # 후보는 승인 대기


def test_gate_failure_is_still_succeeded_and_the_model_is_rejected(tmp_path):
    e = build(tmp_path, trainer=FakeTrainer(accuracy=0.5))
    try:
        e.client.post("/training/runs", headers=e.op)
        e.retrain.wait(20)
        body = e.client.get("/training/runs/RUN-000001", headers=e.op).json()
        assert body["status"] == "succeeded" and body["gate_result"]["passed"] is False and body["result_version"] == "v2"
        assert e.client.get("/models/v2", headers=e.op).json()["stage"] == "rejected"
    finally:
        e.app.state.engine.dispose()


def test_failed_run_shows_the_error_and_no_snapshot(tmp_path):
    e = build(tmp_path, trainer=FakeTrainer(error=RuntimeError("학습 중 터짐")))
    try:
        e.client.post("/training/runs", headers=e.op)
        e.retrain.wait(20)
        body = e.client.get("/training/runs/RUN-000001", headers=e.op).json()
        assert (body["status"], body["step"], body["error_message"], body["result_version"], body["train_data_snapshot"]) == (
            "failed", "train", "학습 중 터짐", None, None)
        assert e.client.get("/models/v2", headers=e.op).status_code == 404
    finally:
        e.app.state.engine.dispose()


def test_list_is_newest_first_with_total_status_filter_and_paging(env):
    post(env)
    finish(env)
    with session_scope(env.app.state.session_factory) as s:  # 풀이 비었으니 다음 실행을 위해 pool 을 더하고, 두 번째 실행
        path = "data/pool/INS-000009.png"
        Image.fromarray(np.full((128, 128), 9, dtype=np.uint8)).save(env.settings.root + "/" + path)
        InspectionRepository(s).add(make_inspection(9, data_split="pool", image_path=path))
    post(env)
    finish(env)
    body = env.client.get("/training/runs", headers=env.op).json()
    assert body["total"] == 2 and [x["run_id"] for x in body["items"]] == ["RUN-000002", "RUN-000001"]
    assert body["items"][0]["train_data_snapshot"]["pool"] == 1 and body["items"][1]["train_data_snapshot"]["pool"] == 2
    assert env.client.get("/training/runs", params={"status": "failed"}, headers=env.op).json() == {"items": [], "total": 0}
    page = env.client.get("/training/runs", params={"limit": 1, "offset": 1}, headers=env.op).json()
    assert page["total"] == 2 and [x["run_id"] for x in page["items"]] == ["RUN-000001"]


@pytest.mark.parametrize("params", [{"status": "done"}, {"status": "대기"}, {"limit": 0}, {"limit": 1001}, {"offset": -1}])
def test_bad_list_query_is_400(env, params):
    r = env.client.get("/training/runs", params=params, headers=env.op)
    assert r.status_code == 400 and r.json()["error"] == "bad_request"


def test_empty_list_has_the_same_shape(env):
    assert env.client.get("/training/runs", headers=env.op).json() == {"items": [], "total": 0}


@pytest.mark.parametrize("run_id", ["RUN-000999", "RUN-1x", "run-000001", "RUN-", "RUN--1", "RUN-٠٠١", "x", "RUN-000001%0A"])
def test_unknown_or_malformed_run_id_is_404(env, run_id):
    post(env)
    finish(env)
    r = env.client.get(f"/training/runs/{run_id}", headers=env.op)
    assert r.status_code == 404 and r.json()["error"] == "not_found"


def test_other_methods_are_405(env):
    assert env.client.delete("/training/runs", headers=env.op).status_code == 405
    assert env.client.put("/training/runs/RUN-000001", headers=env.op).status_code == 405


# ---------- 서버가 켜질 때 정리 ----------

def test_startup_marks_leftover_runs_failed(tmp_path):
    e = build(tmp_path, production=False)  # 운영 모델이 없으면 시작할 때 모델을 불러오지 않는다 (torch 없음)
    with session_scope(e.app.state.session_factory) as s:
        runs = TrainingRunRepository(s)
        runs.add("manual")
        running = runs.add("manual")
        runs.update(running.run_id, status="running", step="train")
    with TestClient(e.app, raise_server_exceptions=False) as client:  # with: 서버가 켜질 때(lifespan) 실행
        body = client.get("/training/runs", headers=e.op).json()
        assert body["total"] == 2 and {x["status"] for x in body["items"]} == {"failed"}
        assert {x["error_message"] for x in body["items"]} == {INTERRUPTED}
        assert client.post("/training/runs", headers=e.op).status_code == 201  # 정리 후에는 새 실행이 가능하다
    e.retrain.wait(20)
