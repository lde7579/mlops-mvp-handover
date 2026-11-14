"""POST /predict: 검사 이미지를 운영 모델로 판정해 prediction 으로 저장한다 + /health 의 loaded."""
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import text

from backend.database import create_db_engine, session_scope
from backend.inference import LOAD_FAILED, NO_MODEL
from backend.main import create_app
from backend.registry import sync_models
from backend.repository import InspectionRepository, ModelVersionRepository, PredictionRepository, Stage
from tests.backend.helpers import (
    asgi_call, bearer, clone_version, copy_models, make_inspection, make_metrics, make_settings, raw_inspection, write_png,
)

PREDICTION_KEYS = {"prediction_id", "inspection_id", "model_version", "predicted_label", "score", "threshold", "latency_ms", "predicted_at"}


class Env:
    def __init__(self, root, trained_models, *, production=True, db_path=":memory:", models=True, images=2):
        self.root, self.models = root, trained_models
        if models:
            copy_models(trained_models.root, root)
        self.settings = make_settings(root=str(root), db_path=str(db_path))
        self.app = create_app(self.settings)
        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.headers = bearer(self.settings, "operator")
        with session_scope(self.app.state.session_factory) as s:
            repo = InspectionRepository(s)
            for n, image in enumerate(trained_models.images[:images], start=1):  # 함수 단위 환경은 2장이면 된다 (INS-000001~2)
                rel = f"data/pool/INS-{n:06d}.png"
                write_png(root, rel, image)
                repo.add(make_inspection(n, image_path=rel, true_label="NG" if n % 2 else "OK", missing_parts=[1] if n % 2 else []))
            if production and models:
                sync_models(s, root / "models", production="v1")

    def predict(self, inspection_id="INS-000001", headers=None, **kw):
        return self.client.post("/predict", json={"inspection_id": inspection_id}, headers=self.headers if headers is None else headers, **kw)

    def prediction_count(self) -> int:
        with session_scope(self.app.state.session_factory) as s:
            return PredictionRepository(s).count()

    def health(self):
        return self.client.get("/health").json()["model"]

    def close(self):
        self.app.state.engine.dispose()


@pytest.fixture(scope="module")
def shared(tmp_path_factory, trained_models):
    e = Env(tmp_path_factory.mktemp("predict_shared"), trained_models, images=6)  # 읽기만 하는 시험이 INS-000001~6 을 쓴다
    yield e
    e.close()


@pytest.fixture
def env(tmp_path, trained_models):
    e = Env(tmp_path, trained_models)
    yield e
    e.close()


# ---------- 인증 ----------

def test_requires_login(shared):
    for headers in ({}, {"Authorization": "Bearer garbage"}):
        r = shared.client.post("/predict", json={"inspection_id": "INS-000001"}, headers=headers)
        assert r.status_code == 401 and r.json()["error"] == "unauthorized" and r.headers["www-authenticate"] == "Bearer"


def test_401_comes_before_body_validation(shared):
    assert shared.client.post("/predict", json={"nope": 1}).status_code == 401


@pytest.mark.parametrize("user", ["operator", "admin"])
def test_operator_and_admin_can_predict(shared, user):
    assert shared.predict(headers=bearer(shared.settings, user)).status_code == 201


# ---------- 응답과 저장 ----------

def test_created_response_has_the_8_contract_fields(shared):
    r = shared.predict("INS-000002")
    body = r.json()
    assert r.status_code == 201 and set(body) == PREDICTION_KEYS
    assert body["inspection_id"] == "INS-000002" and body["model_version"] == "v1"
    assert body["prediction_id"].startswith("PRD-") and len(body["prediction_id"]) == 10
    assert body["predicted_label"] in ("OK", "NG") and 0.0 <= body["score"] <= 1.0
    assert body["threshold"] == shared.models.metrics.threshold == 0.5  # 불러온 모델에 기록된 값
    assert body["predicted_label"] == ("NG" if body["score"] >= body["threshold"] else "OK")
    assert isinstance(body["latency_ms"], int) and 0 <= body["latency_ms"] <= 5000
    assert body["predicted_at"].endswith("Z") and abs(
        (datetime.now(timezone.utc) - datetime.fromisoformat(body["predicted_at"])).total_seconds()) < 60


def test_prediction_is_saved_and_shown_in_the_inspection_detail(shared):
    body = shared.predict("INS-000003").json()
    detail = shared.client.get("/inspections/INS-000003", headers=shared.headers).json()
    assert body in detail["predictions"]
    with session_scope(shared.app.state.session_factory) as s:
        saved = PredictionRepository(s).get(body["prediction_id"])
    assert saved is not None and saved.score == body["score"] and saved.model_version == "v1"


def test_score_matches_the_model_used_directly(shared):
    from parts.predictor import Predictor

    direct = Predictor(root=shared.root)
    direct.load("v1")
    for n in (1, 4):
        r = shared.predict(f"INS-{n:06d}").json()
        assert r["score"] == direct.predict(shared.models.images[n - 1]).score  # 같은 전처리·같은 모델


def test_same_inspection_can_be_predicted_many_times(shared):
    first, second = shared.predict("INS-000005").json(), shared.predict("INS-000005").json()
    assert first["prediction_id"] != second["prediction_id"] and first["score"] == second["score"]
    predictions = shared.client.get("/inspections/INS-000005", headers=shared.headers).json()["predictions"]
    ids = [p["prediction_id"] for p in predictions]
    assert ids[:2] == [second["prediction_id"], first["prediction_id"]]  # 최신 먼저, 이전 판정은 그대로


def test_prediction_ids_are_unique_and_increasing(shared):
    ids = [shared.predict(f"INS-{n:06d}").json()["prediction_id"] for n in (1, 2, 3)]
    assert ids == sorted(ids) and len(set(ids)) == 3


def test_a_single_request_finishes_well_within_5_seconds(shared):
    start = time.perf_counter()
    r = shared.predict("INS-000006")
    assert r.status_code == 201 and time.perf_counter() - start < 5


def test_row_is_committed_before_the_response_starts(tmp_path, trained_models):
    e = Env(tmp_path, trained_models, db_path=tmp_path / "test.db")
    other = create_db_engine(tmp_path / "test.db")
    seen = []
    try:
        def on_start(status):
            with other.connect() as conn:
                seen.append((status, conn.exec_driver_sql("SELECT COUNT(*) FROM prediction").scalar()))

        asgi_call(e.app, "POST", "/predict", on_start, headers={**e.headers, "content-type": "application/json"},
                  body=b'{"inspection_id": "INS-000001"}')
        assert seen == [(201, 1)]
    finally:
        other.dispose()
        e.close()


def test_concurrent_requests_save_unique_predictions(tmp_path, trained_models):
    e = Env(tmp_path, trained_models, db_path=tmp_path / "test.db", images=4)  # 4개 검사를 판정한다
    try:
        with ThreadPoolExecutor(4) as pool:
            batches = list(pool.map(lambda n: [e.predict(f"INS-{n:06d}") for _ in range(3)], range(1, 5)))
        responses = [r for batch in batches for r in batch]
        assert [r.status_code for r in responses] == [201] * 12
        ids = sorted(r.json()["prediction_id"] for r in responses)
        assert ids == [f"PRD-{n:06d}" for n in range(1, 13)] and e.prediction_count() == 12
    finally:
        e.close()


# ---------- 요청 오류 ----------

@pytest.mark.parametrize("payload", [
    {}, {"inspection_id": ""}, {"inspection_id": 1}, {"inspection_id": None}, {"inspection_id": "x" * 65},
    {"inspection_id": "INS-000001", "model_version": "v1"}, {"inspection": "INS-000001"},
])
def test_bad_body_is_400_and_saves_nothing(shared, payload):
    before = shared.prediction_count()
    r = shared.client.post("/predict", json=payload, headers=shared.headers)
    assert r.status_code == 400 and r.json()["error"] == "bad_request" and set(r.json()) == {"error", "message"}
    assert shared.prediction_count() == before


@pytest.mark.parametrize("content", [b"", b"not json", b"[1]", b'"INS-000001"'])
def test_non_object_body_is_400(shared, content):
    r = shared.client.post("/predict", content=content, headers={**shared.headers, "Content-Type": "application/json"})
    assert r.status_code == 400 and r.json()["error"] == "bad_request"


def test_extra_field_message_names_the_field(shared):
    r = shared.client.post("/predict", json={"inspection_id": "INS-000001", "threshold": 0.1}, headers=shared.headers)
    assert "threshold" in r.json()["message"]


def test_other_methods_are_405(shared):
    r = shared.client.get("/predict", headers=shared.headers)
    assert r.status_code == 405 and "POST" in r.headers["allow"]


def test_unknown_inspection_is_404_and_saves_nothing(shared):
    before = shared.prediction_count()
    r = shared.predict("INS-999999")
    assert r.status_code == 404 and r.json() == {"error": "not_found", "message": "검사를 찾을 수 없습니다"}
    assert shared.prediction_count() == before


# ---------- 이미지 문제 ----------

def add_row(env, n, path):
    with session_scope(env.app.state.session_factory) as s:
        raw_inspection(s.connection(), id=f"INS-{n:06d}", path=path)


@pytest.mark.parametrize("path", ["data/pool/gone.png", "../outside.png", "data/../parts/collector/collector.toml",
                                  "data/mlops.db", "parts/collector/collector.toml"])
def test_missing_or_forbidden_image_path_is_404_and_saves_nothing(env, path):
    add_row(env, 50, path)
    r = env.predict("INS-000050")
    assert r.status_code == 404 and r.json() == {"error": "not_found", "message": "이미지를 찾을 수 없습니다"}
    assert env.prediction_count() == 0


@pytest.mark.parametrize("make", [
    lambda p: Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(p),            # 크기가 다르다
    lambda p: Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(p),       # 흑백이 아니다
    lambda p: p.write_bytes(b"this is not a png"),                                     # 손상된 파일
])
def test_image_that_breaks_the_rules_is_500_without_details_and_saves_nothing(env, make):
    target = env.root / "data" / "pool" / "bad.png"
    make(target)
    add_row(env, 51, "data/pool/bad.png")
    r = env.predict("INS-000051")
    assert r.status_code == 500 and r.json()["error"] == "server_error"
    assert "bad.png" not in r.text and str(env.root) not in r.text
    assert env.prediction_count() == 0


# ---------- 모델을 쓸 수 없을 때: 503 ----------

def test_no_production_model_is_503(tmp_path, trained_models):
    e = Env(tmp_path, trained_models, production=False)
    try:
        r = e.predict()
        assert r.status_code == 503 and r.json() == {"error": "service_unavailable", "message": NO_MODEL}
        assert e.prediction_count() == 0 and e.health() == {"production_version": None, "loaded": False}
    finally:
        e.close()


def test_candidate_only_is_503(tmp_path, trained_models):
    e = Env(tmp_path, trained_models, production=False)
    try:
        with session_scope(e.app.state.session_factory) as s:
            sync_models(s, tmp_path / "models")  # v1 = candidate
        assert e.predict().json()["message"] == NO_MODEL
    finally:
        e.close()


def test_production_without_model_files_is_503_without_leaking_paths(env):
    import shutil
    shutil.rmtree(env.root / "models" / "v1")
    r = env.predict()
    assert r.status_code == 503 and r.json() == {"error": "service_unavailable", "message": LOAD_FAILED}
    assert "model.pt" not in r.text and str(env.root) not in r.text and env.prediction_count() == 0
    assert env.health() == {"production_version": "v1", "loaded": False}


def test_broken_model_is_503_every_time_and_recovers_when_fixed_without_restart(env, caplog):
    model = env.root / "models" / "v1" / "model.pt"
    good = model.read_bytes()
    model.write_bytes(b"broken")
    with caplog.at_level(logging.WARNING, logger="backend"):
        for _ in range(3):
            assert env.predict().status_code == 503
    assert sum(1 for r in caplog.records if r.getMessage().startswith("운영 모델 ")) == 1  # 같은 실패는 로그 한 번
    assert env.prediction_count() == 0 and env.health()["loaded"] is False
    model.write_bytes(good)  # 파일을 고치면 서버를 다시 켜지 않아도 된다
    r = env.predict()
    assert r.status_code == 201 and env.health() == {"production_version": "v1", "loaded": True}


# ---------- /health 의 loaded ----------

def test_health_loaded_becomes_true_after_the_first_prediction(env):
    assert env.health() == {"production_version": "v1", "loaded": False}  # 건강 확인은 모델을 불러오지 않는다
    assert env.health() == {"production_version": "v1", "loaded": False}
    assert env.predict().status_code == 201
    assert env.health() == {"production_version": "v1", "loaded": True}


def test_health_loaded_is_false_when_production_moves_to_a_version_not_loaded_yet(env):
    clone_version(env.root, "v1", "v2")
    with session_scope(env.app.state.session_factory) as s:
        ModelVersionRepository(s).register(make_metrics("v2", created_minutes=1), stage=Stage.CANDIDATE)
    env.predict()
    assert env.health() == {"production_version": "v1", "loaded": True}
    with session_scope(env.app.state.session_factory) as s:
        s.execute(text("UPDATE model_version SET stage='archived' WHERE version='v1'"))
        s.execute(text("UPDATE model_version SET stage='production' WHERE version='v2'"))
    assert env.health() == {"production_version": "v2", "loaded": False}  # 운영은 v2, 불러온 것은 아직 v1
    r = env.predict()
    assert r.json()["model_version"] == "v2"  # 다음 판정부터 새 운영 모델
    assert env.health() == {"production_version": "v2", "loaded": True}


def test_new_production_uses_the_threshold_recorded_with_that_model(env):
    clone_version(env.root, "v1", "v2", threshold=0.9)
    with session_scope(env.app.state.session_factory) as s:
        ModelVersionRepository(s).register(make_metrics("v2", created_minutes=1, threshold=0.9), stage=Stage.CANDIDATE)
    assert env.predict().json()["threshold"] == 0.5
    with session_scope(env.app.state.session_factory) as s:
        s.execute(text("UPDATE model_version SET stage='archived' WHERE version='v1'"))
        s.execute(text("UPDATE model_version SET stage='production' WHERE version='v2'"))
    body = env.predict().json()
    assert body["model_version"] == "v2" and body["threshold"] == 0.9
    assert body["predicted_label"] == ("NG" if body["score"] >= 0.9 else "OK")


def test_health_without_any_model_does_not_load_anything(tmp_path, trained_models):
    e = Env(tmp_path, trained_models, production=False)
    try:
        assert e.client.get("/health").json()["model"] == {"production_version": None, "loaded": False}
        assert e.app.state.inference.loaded_version is None
    finally:
        e.close()


# ---------- 문서 ----------

def test_predict_and_503_are_in_openapi(shared):
    spec = shared.client.get("/openapi.json").json()
    responses = spec["paths"]["/predict"]["post"]["responses"]
    assert "201" in responses and "503" in responses and "422" not in responses and "400" in responses


def test_503_error_name_is_service_unavailable():
    from backend.errors import DEFAULT_MESSAGES, ERROR_NAMES, error_name

    assert ERROR_NAMES[503] == error_name(503) == "service_unavailable" and 503 in DEFAULT_MESSAGES
