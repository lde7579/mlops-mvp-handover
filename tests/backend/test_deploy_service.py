"""승인 중 판정과 동시성 (DeployService + InferenceService, api-v1.md 4장 "승인 중 판정").
파일 DB(WAL)와 스레드를 쓴다. 기다림은 sleep 이 아니라 Event 로 맞춘다. 마지막 시험 하나만 실제 모델(torch)을 쓴다."""
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.database import session_scope
from backend.deployment import DeployService
from backend.inference import InferenceService
from backend.main import create_app
from backend.repository import (
    ApprovalRepository,
    AuditLogRepository,
    InspectionRepository,
    ModelVersionRepository,
    PredictionRepository,
    Stage,
    TrainingRunRepository,
)
from tests.backend.helpers import (
    FakeParts,
    at,
    bearer,
    clone_version,
    copy_models,
    make_inspection,
    make_metrics,
    make_settings,
    write_png,
)


def build(tmp_path, *, staging=("v2",), fake=True, threshold=None):
    """파일 DB(여러 스레드가 함께 쓴다), v1 production, staging 후보와 승인 요청, 검사 1건. fake=False 면 실제 추론부를 쓴다."""
    settings = make_settings(root=str(tmp_path), db_path=str(tmp_path / "test.db"))
    app = create_app(settings)
    parts = FakeParts() if fake else None
    if fake:
        app.state.inference = InferenceService(settings, parts_loader=parts)
        app.state.deploy = DeployService(app.state.session_factory, app.state.inference)
    extra = {} if threshold is None else {"threshold": threshold}
    with session_scope(app.state.session_factory) as s:
        models, approvals = ModelVersionRepository(s), ApprovalRepository(s)
        models.register(make_metrics("v1", **extra), stage=Stage.PRODUCTION)
        for n, version in enumerate(staging, start=1):
            models.register(make_metrics(version, created_minutes=n, **extra), stage=Stage.STAGING)
            approvals.add_pending(version, requested_by="system", created_at=at(n))
        InspectionRepository(s).add(make_inspection(1))
    return SimpleNamespace(app=app, settings=settings, fake=parts, inference=app.state.inference,
                           client=TestClient(app, raise_server_exceptions=False),
                           op=bearer(settings, "operator"), admin=bearer(settings, "admin"))


def predict(env):
    return env.client.post("/predict", json={"inspection_id": "INS-000001"}, headers=env.op)


def approve(env, approval_id="APR-000001"):
    return env.client.post(f"/approvals/{approval_id}/approve", json={}, headers=env.admin)


@pytest.fixture
def env(tmp_path):
    write_png(tmp_path, "data/pool/INS-000001.png", np.zeros((128, 128), dtype=np.uint8))
    e = build(tmp_path)
    yield e
    e.app.state.engine.dispose()


def wait_for(condition, what, timeout=10):
    tick, deadline = threading.Event(), time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        tick.wait(0.002)
    assert condition(), f"기다리던 일이 일어나지 않았습니다: {what}"


# ---------- 승인 중 연속 판정 ----------

def test_predictions_keep_working_while_the_new_model_is_read_and_switch_exactly_once(env):
    """새 모델을 읽는 동안에도 /predict 는 막히지 않고(DB 쓰기 잠금을 쥐지 않는다) 오류가 없다. 교체 뒤에는 모두 v2 다."""
    reading, release = threading.Event(), threading.Event()

    def hold_while_reading_v2(version):
        if version == "v2":
            reading.set()
            release.wait(10)

    env.fake.on_load = hold_while_reading_v2
    assert predict(env).status_code == 201  # v1 을 올려 둔다
    results, stop, answered = [], threading.Event(), threading.Event()  # answered: 승인 응답을 받았다

    def predict_loop():
        while not stop.is_set():
            started_after_answer = answered.is_set()  # 요청을 보내기 전에 정한다 (이미 시작한 판정은 이전 모델로 끝날 수 있다)
            r = predict(env)
            results.append((r.status_code, r.json().get("model_version"), started_after_answer))

    looper = threading.Thread(target=predict_loop)
    answer = {}
    approver = threading.Thread(target=lambda: answer.update(response=approve(env)))
    looper.start()
    try:
        wait_for(lambda: len(results) >= 3, "승인 전 판정")
        approver.start()
        assert reading.wait(10), "새 모델 읽기가 시작되지 않았습니다"
        during = len(results)
        wait_for(lambda: len(results) >= during + 5, "읽는 동안의 판정 (막히면 여기서 실패한다)")
        release.set()
        approver.join(10)
        assert answer["response"].status_code == 200
        answered.set()
        returned_at = len(results)
        wait_for(lambda: len(results) >= returned_at + 5, "승인 뒤의 판정")
    finally:
        release.set()
        stop.set()
        looper.join(10)
    codes = [code for code, _, _ in results]
    versions = [v for _, v, _ in results]
    assert set(codes) == {201}, f"오류가 있었습니다: {sorted(set(codes))}"
    first_v2 = versions.index("v2")
    assert set(versions[:first_v2]) == {"v1"} and set(versions[first_v2:]) == {"v2"}  # v1...v1 v2...v2, 한 번만 바뀐다
    assert any(late for _, _, late in results)
    assert all(v == "v2" for _, v, late in results if late)  # 응답을 받은 뒤에 시작한 판정은 모두 새 버전
    with session_scope(env.app.state.session_factory) as s:
        stored = PredictionRepository(s).list(limit=1000)
    assert {p.model_version for p in stored} == {"v1", "v2"}
    assert all(p.threshold == (0.5 if p.model_version == "v1" else 0.7) for p in stored)  # 모델과 임계값이 섞이지 않았다


# ---------- 동시에 두 번 눌렀을 때 ----------

def test_the_same_request_approved_twice_at_once_succeeds_once(env):
    with ThreadPoolExecutor(2) as pool:
        codes = sorted(f.result().status_code for f in [pool.submit(approve, env), pool.submit(approve, env)])
    assert codes == [200, 409]
    assert env.fake.loads.count("v2") == 1  # 늦게 온 쪽은 모델을 읽지 않았다
    with session_scope(env.app.state.session_factory) as s:
        assert ModelVersionRepository(s).get_production().version == "v2"


def test_two_candidates_approved_at_once_only_the_first_gets_in(tmp_path):
    write_png(tmp_path, "data/pool/INS-000001.png", np.zeros((128, 128), dtype=np.uint8))
    env = build(tmp_path, staging=("v2", "v3"))
    try:
        with session_scope(env.app.state.session_factory) as s:
            runs = TrainingRunRepository(s)
            for version in ("v2", "v3"):
                run = runs.add("manual")
                runs.update(run.run_id, status="succeeded", result_version=version,
                            gate_result={"baseline_version": "v1", "passed": True, "checks": []})
        with ThreadPoolExecutor(2) as pool:
            codes = sorted(f.result().status_code for f in [pool.submit(approve, env, "APR-000001"), pool.submit(approve, env, "APR-000002")])
        assert codes == [200, 409]  # 나중에 처리된 쪽은 기준 모델이 바뀌어 거부된다
        with session_scope(env.app.state.session_factory) as s:
            assert ModelVersionRepository(s).get_production().version in ("v2", "v3")
    finally:
        env.app.state.engine.dispose()


# ---------- 실제 모델 (이 파일의 유일한 torch 사용) ----------

def test_real_predictor_is_replaced_after_approval_and_a_missing_rollback_folder_is_503(tmp_path, trained_models):
    """가짜로는 증명할 수 없는 것: 진짜 Predictor 두 개가 상태를 공유하지 않고 교체되고, 새 판정이 v2 로 기록된다. 모델 폴더가 없을 때의 진짜 읽기 실패."""
    copy_models(trained_models.root, tmp_path)
    clone_version(tmp_path, "v1", "v2")  # 폴더를 복사하고 metrics.json 의 version 만 v2 로 고친다 (가중치는 같다)
    assert json.loads((tmp_path / "models" / "v2" / "metrics.json").read_text(encoding="utf-8"))["version"] == "v2"
    threshold = json.loads((tmp_path / "models" / "v1" / "metrics.json").read_text(encoding="utf-8"))["threshold"]
    write_png(tmp_path, "data/pool/INS-000001.png", trained_models.images[0])
    env = build(tmp_path, fake=False, threshold=threshold)
    try:
        first = predict(env)
        assert first.status_code == 201 and first.json()["model_version"] == "v1"
        old_predictor = env.inference._predictor
        r = approve(env)
        assert r.status_code == 200 and r.json()["reload"]["loaded_version"] == "v2"
        assert env.inference._predictor is not old_predictor and old_predictor.version == "v1"  # 이전 객체는 그대로, 새 객체로 교체
        assert env.inference.loaded_version == "v2"
        after = predict(env)
        assert after.status_code == 201 and after.json()["model_version"] == "v2" and after.json()["threshold"] == threshold
        assert env.client.get("/health").json()["model"] == {"production_version": "v2", "loaded": True}
        # 롤백할 모델 폴더(models/v1)가 지워졌다: 진짜 파일 읽기 실패도 503 이고, 운영 모델은 v2 그대로다
        shutil.rmtree(tmp_path / "models" / "v1")
        failed = env.client.post("/models/v1/rollback", json={"reason": "성능 저하"}, headers=env.admin)
        assert failed.status_code == 503 and failed.json()["error"] == "service_unavailable"
        assert env.client.get("/models/production", headers=env.op).json()["version"] == "v2"
        assert env.inference.loaded_version == "v2" and predict(env).json()["model_version"] == "v2"
        with session_scope(env.app.state.session_factory) as s:
            assert AuditLogRepository(s).list(limit=1)[0].action.value == "rollback_fail"
    finally:
        env.app.state.engine.dispose()
