"""POST /simulator/start·stop·drift, GET /simulator/status (api-v1.md A23). 메모리 DB, 가짜 생성·판정(DB·수집기·torch 없음).

앱의 러너를 가짜를 넣은 러너로 바꿔 끼운다 (라우터와 서버 종료 처리는 app.state.simulator 를 쓴다).
진짜 수집기·파일 DB 와의 연결은 수동 확인에서 본다 (메모리 DB 는 연결 하나를 스레드가 공유하기 때문).
"""
import pytest
from fastapi.testclient import TestClient

from backend.database import session_scope
from backend.inference import InferenceService
from backend.main import create_app
from backend.repository import ModelVersionRepository, Stage
from backend.simulator_runner import SimulatorRunner, build_runner
from tests.backend.helpers import FakeParts, bearer, make_metrics, make_settings
from tests.backend.test_simulator_runner import WAIT_LIMIT, Fakes

STATUS_KEYS = {"running", "interval_seconds", "defect_rate", "drift", "generated", "judged", "judge_failures", "last_error", "started_at"}


class Env:
    def __init__(self):
        self.settings = make_settings()
        self.app = create_app(self.settings)
        self.fakes = Fakes(cycles=2)
        self.runner = SimulatorRunner(self.fakes.generate, self.fakes.judge, wait=self.fakes.wait)
        self.app.state.simulator = self.runner
        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.op, self.admin = bearer(self.settings, "operator"), bearer(self.settings, "admin")

    def close(self):
        self.runner.shutdown()
        assert self.runner.thread is None or not self.runner.thread.is_alive(), "시험이 끝났는데 자동 생성 스레드가 남아 있습니다"
        self.app.state.engine.dispose()


@pytest.fixture
def env():
    e = Env()
    yield e
    e.close()


def parked(env):
    assert env.fakes.parked.wait(WAIT_LIMIT)


# ---------- 권한·형식 ----------

@pytest.mark.parametrize("method,path", [("post", "/simulator/start"), ("post", "/simulator/stop"),
                                         ("post", "/simulator/drift"), ("get", "/simulator/status")])
def test_requires_login(env, method, path):
    r = env.client.request(method.upper(), path, json={"enabled": False} if path.endswith("drift") else None)
    assert r.status_code == 401 and r.json()["error"] == "unauthorized" and r.headers["www-authenticate"] == "Bearer"


def test_admin_can_control_too(env):
    assert env.client.post("/simulator/start", headers=env.admin).status_code == 200
    assert env.client.post("/simulator/stop", headers=env.admin).status_code == 200


@pytest.mark.parametrize("method,path", [("get", "/simulator/start"), ("get", "/simulator/stop"),
                                         ("get", "/simulator/drift"), ("post", "/simulator/status")])
def test_wrong_method_is_405_with_allow(env, method, path):
    r = getattr(env.client, method)(path, headers=env.op)
    assert r.status_code == 405 and "allow" in r.headers


def test_new_paths_are_in_openapi(env):
    paths = env.client.get("/openapi.json").json()["paths"]
    for path in ("/simulator/start", "/simulator/stop", "/simulator/drift", "/simulator/status"):
        assert path in paths


# ---------- status ----------

def test_status_before_anything_is_stopped_with_nulls(env):
    r = env.client.get("/simulator/status", headers=env.op)
    assert r.status_code == 200
    assert r.json() == {"running": False, "interval_seconds": None, "defect_rate": None, "drift": {"enabled": False, "drift_type": None},
                        "generated": 0, "judged": 0, "judge_failures": 0, "last_error": None, "started_at": None}


def test_create_app_does_not_start_the_runner():
    e = create_app(make_settings())
    try:
        assert e.state.simulator.status().running is False and e.state.simulator.thread is None
    finally:
        e.state.engine.dispose()


# ---------- start ----------

def test_start_without_body_uses_defaults(env):
    r = env.client.post("/simulator/start", headers=env.op)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == STATUS_KEYS
    assert (body["running"], body["interval_seconds"], body["defect_rate"]) == (True, 2.0, 0.2)
    assert body["started_at"].endswith("Z")


def test_start_with_settings_and_status_reflects_counts(env):
    r = env.client.post("/simulator/start", json={"defect_rate": 0.5, "interval_seconds": 1.5}, headers=env.op)
    assert r.status_code == 200 and (r.json()["interval_seconds"], r.json()["defect_rate"]) == (1.5, 0.5)
    parked(env)
    s = env.client.get("/simulator/status", headers=env.op).json()
    assert (s["running"], s["generated"], s["judged"], s["judge_failures"]) == (True, 2, 2, 0)
    assert env.fakes.waits == [1.5, 1.5]


def test_start_while_running_is_409_and_keeps_the_first_settings(env):
    env.client.post("/simulator/start", json={"defect_rate": 0.3, "interval_seconds": 1.0}, headers=env.op)
    r = env.client.post("/simulator/start", json={"defect_rate": 0.9}, headers=env.op)
    assert r.status_code == 409 and r.json() == {"error": "conflict", "message": "이미 실행 중입니다"}
    s = env.client.get("/simulator/status", headers=env.op).json()
    assert (s["interval_seconds"], s["defect_rate"]) == (1.0, 0.3)


@pytest.mark.parametrize("body", [
    {"interval_seconds": 0.9}, {"interval_seconds": 3.1}, {"defect_rate": -0.01}, {"defect_rate": 1.01},
    {"defect_rate": "0.5"}, {"interval_seconds": "2"}, {"defect_rate": True}, {"defect_rate": None}, {"speed": 1}, [],
])
def test_start_rejects_bad_requests_with_400(env, body):
    r = env.client.post("/simulator/start", json=body, headers=env.op)
    assert r.status_code == 400 and r.json()["error"] == "bad_request"
    assert env.client.get("/simulator/status", headers=env.op).json()["running"] is False  # 시작되지 않았다


def test_start_accepts_boundary_values_and_integers(env):
    r = env.client.post("/simulator/start", json={"defect_rate": 0, "interval_seconds": 3}, headers=env.op)
    assert r.status_code == 200 and (r.json()["defect_rate"], r.json()["interval_seconds"]) == (0.0, 3.0)


def test_start_works_even_when_judging_is_impossible(env):
    from backend.inference import ModelUnavailableError

    env.fakes.judge_error = ModelUnavailableError("운영 중인 모델이 없습니다")
    assert env.client.post("/simulator/start", headers=env.op).status_code == 200
    parked(env)
    s = env.client.get("/simulator/status", headers=env.op).json()
    assert (s["running"], s["generated"], s["judged"], s["judge_failures"]) == (True, 2, 0, 2)
    assert s["last_error"] == "판정 건너뜀: 운영 중인 모델이 없습니다"


# ---------- stop ----------

def test_stop_ends_the_thread_and_answers_with_stopped_status(env):
    env.client.post("/simulator/start", headers=env.op)
    parked(env)
    r = env.client.post("/simulator/stop", headers=env.op)
    assert r.status_code == 200 and r.json()["running"] is False and r.json()["generated"] == 2
    assert not env.runner.thread.is_alive()  # 응답을 받은 시점에 스레드가 끝나 있다


def test_stop_when_not_running_is_200(env):
    for _ in range(2):
        r = env.client.post("/simulator/stop", headers=env.op)
        assert r.status_code == 200 and r.json()["running"] is False


def test_can_start_again_after_stop(env):
    env.client.post("/simulator/start", headers=env.op)
    parked(env)
    env.client.post("/simulator/stop", headers=env.op)
    env.fakes.parked.clear(), env.fakes.waits.clear()
    assert env.client.post("/simulator/start", headers=env.op).status_code == 200


# ---------- drift ----------

def test_drift_on_then_off(env):
    r = env.client.post("/simulator/drift", json={"enabled": True, "drift_type": "dark"}, headers=env.op)
    assert r.status_code == 200 and set(r.json()) == STATUS_KEYS
    assert r.json()["drift"] == {"enabled": True, "drift_type": "dark"} and r.json()["running"] is False  # 꺼진 상태에서도 된다
    assert env.client.get("/simulator/status", headers=env.op).json()["drift"] == {"enabled": True, "drift_type": "dark"}
    r = env.client.post("/simulator/drift", json={"enabled": False}, headers=env.op)
    assert r.status_code == 200 and r.json()["drift"] == {"enabled": False, "drift_type": None}


def test_drift_type_can_be_changed_while_on(env):
    env.client.post("/simulator/drift", json={"enabled": True, "drift_type": "dark"}, headers=env.op)
    r = env.client.post("/simulator/drift", json={"enabled": True, "drift_type": "shift"}, headers=env.op)
    assert r.json()["drift"] == {"enabled": True, "drift_type": "shift"}


def test_running_generation_uses_the_drift_set_through_the_api(env):
    env.client.post("/simulator/drift", json={"enabled": True, "drift_type": "noise"}, headers=env.op)
    env.client.post("/simulator/start", headers=env.op)
    parked(env)
    assert [(s, d) for s, _, d in env.fakes.calls] == [("drift", "noise")] * 2


@pytest.mark.parametrize("body", [
    {"enabled": True}, {"enabled": True, "drift_type": None}, {"enabled": False, "drift_type": "dark"},
    {"enabled": True, "drift_type": "bright"}, {"enabled": True, "drift_type": "DARK"}, {"drift_type": "dark"},
    {"enabled": "true", "drift_type": "dark"}, {"enabled": 1, "drift_type": "dark"}, {"enabled": True, "drift_type": "dark", "strength": 1},
    {},
])
def test_drift_rejects_bad_requests_with_400(env, body):
    r = env.client.post("/simulator/drift", json=body, headers=env.op)
    assert r.status_code == 400 and r.json()["error"] == "bad_request"
    assert env.client.get("/simulator/status", headers=env.op).json()["drift"] == {"enabled": False, "drift_type": None}  # 바뀌지 않았다


def test_drift_without_body_is_400(env):
    assert env.client.post("/simulator/drift", headers=env.op).status_code == 400


# ---------- 서버 종료 ----------

def test_server_shutdown_stops_the_running_thread():
    e = Env()
    try:
        with TestClient(e.app, raise_server_exceptions=False) as client:  # with: 서버가 켜지고(lifespan) 꺼질 때까지
            assert client.post("/simulator/start", headers=e.op).status_code == 200
            parked(e)
            assert e.runner.thread.is_alive()
        assert not e.runner.thread.is_alive() and e.runner.status().running is False
    finally:
        e.runner.shutdown()
        e.app.state.engine.dispose()


def test_server_restart_starts_stopped_with_drift_off():
    """새 앱(= 서버를 다시 켠 것)은 자동 생성이 멈춘 상태이고 드리프트 주입은 꺼진 상태다 (메모리 상태)."""
    first = Env()
    first.client.post("/simulator/drift", json={"enabled": True, "drift_type": "dark"}, headers=first.op)
    first.close()
    second = create_app(make_settings())
    try:
        s = second.state.simulator.status()
        assert (s.running, s.drift_enabled, s.drift_type) == (False, False, None)
    finally:
        second.state.engine.dispose()


# ---------- 진짜 생성기·판정 연결 (build_runner): 수집기가 만든 이미지를 파일 DB 에 넣고 가짜 모델로 판정 ----------

def test_real_generation_and_judging_are_wired_through_build_runner(tmp_path):
    settings = make_settings(root=str(tmp_path), db_path=str(tmp_path / "wired.db"))
    app = create_app(settings)
    fakes = Fakes(cycles=3)
    app.state.simulator = build_runner(settings, app.state.session_factory, app.state.generation,
                                       InferenceService(settings, parts_loader=FakeParts()), wait=fakes.wait)
    with session_scope(app.state.session_factory) as s:
        ModelVersionRepository(s).register(make_metrics("v1"), stage=Stage.PRODUCTION)
    client, op = TestClient(app, raise_server_exceptions=False), bearer(settings, "operator")
    try:
        assert client.post("/simulator/drift", json={"enabled": True, "drift_type": "dark"}, headers=op).status_code == 200
        assert client.post("/simulator/start", json={"defect_rate": 1.0, "interval_seconds": 1.0}, headers=op).status_code == 200
        assert fakes.parked.wait(WAIT_LIMIT)
        status = client.get("/simulator/status", headers=op).json()
        assert (status["generated"], status["judged"], status["judge_failures"], status["last_error"]) == (3, 3, 0, None)

        items = client.get("/inspections", params={"limit": 10}, headers=op).json()
        assert items["total"] == 3
        assert {(i["data_split"], i["scenario"], i["drift_type"], i["true_label"]) for i in items["items"]} == {("pool", "drift", "dark", "NG")}
        assert len(list((tmp_path / "data" / "pool").glob("*.png"))) == 3  # 이미지 파일도 3장

        predictions = client.get("/predictions", headers=op).json()
        assert predictions["total"] == 3
        assert {(p["model_version"], p["threshold"]) for p in predictions["items"]} == {("v1", 0.5)}
        assert {p["inspection_id"] for p in predictions["items"]} == {i["inspection_id"] for i in items["items"]}  # 검사마다 판정 1건
    finally:
        app.state.simulator.shutdown()
        app.state.engine.dispose()
