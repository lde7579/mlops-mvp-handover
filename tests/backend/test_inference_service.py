"""backend/inference.py: 운영 모델 불러오기, 판정, 실패 시 재시도와 복구, 서버 시작 때 미리 불러오기."""
import json
import logging
import threading

import pytest
from fastapi.testclient import TestClient

from backend.database import make_session_factory, session_scope
from backend.inference import LOAD_FAILED, NO_MODEL, InferenceService, ModelUnavailableError
from backend.main import create_app
from backend.registry import sync_models
from backend.repository import ModelVersionRepository, Stage
from tests.backend.helpers import clone_version, copy_models, make_metrics, make_settings



def failure_logs(caplog) -> int:
    """'운영 모델 vN 을 불러오지 못했습니다' 경고 기록 수 (오류 문구 안에 같은 말이 또 있어서 기록 단위로 센다)."""
    return sum(1 for r in caplog.records if r.name == "backend" and r.getMessage().startswith("운영 모델 "))


class Env:
    """임시 폴더(models/v1 복사본) + 메모리 DB. production 은 register 로 정한다."""

    def __init__(self, root, factory, trained_models):
        self.root, self.factory, self.models = root, factory, trained_models
        copy_models(trained_models.root, root)
        self.service = InferenceService(make_settings(root=str(root)))

    def register(self, version="v1", stage=Stage.PRODUCTION, **over):
        path = self.root / "models" / version / "metrics.json"
        metrics = make_metrics(version, **{"threshold": json.loads(path.read_text(encoding="utf-8"))["threshold"], **over})
        with session_scope(self.factory) as s:
            ModelVersionRepository(s).register(metrics, stage=stage)

    def set_production(self, old, new):
        with session_scope(self.factory) as s:
            repo = ModelVersionRepository(s)
            s.execute(__import__("sqlalchemy").text("UPDATE model_version SET stage='archived' WHERE version=:v"), {"v": old})
            s.execute(__import__("sqlalchemy").text("UPDATE model_version SET stage='production' WHERE version=:v"), {"v": new})

    def ensure(self):
        with session_scope(self.factory) as s:
            return self.service.ensure_loaded(s)

    def predict(self, i=0):
        with session_scope(self.factory) as s:
            return self.service.predict(s, self.models.images[i])

    def break_files(self, version="v1"):
        (self.root / "models" / version / "model.pt").write_bytes(b"not a model")

    def fix_files(self, version="v1"):
        (self.root / "models" / version / "model.pt").write_bytes((self.models.root / "models" / "v1" / "model.pt").read_bytes())


@pytest.fixture
def env(tmp_path, factory, trained_models):
    return Env(tmp_path, factory, trained_models)


# ---------- 불러오기 ----------

def test_nothing_is_loaded_at_creation(env):
    assert env.service.loaded_version is None


def test_no_production_model_is_unavailable(env):
    with pytest.raises(ModelUnavailableError, match=NO_MODEL):
        env.ensure()
    env.register("v1", Stage.CANDIDATE)  # candidate 는 운영 모델이 아니다
    with pytest.raises(ModelUnavailableError, match=NO_MODEL):
        env.ensure()
    assert env.service.loaded_version is None


def test_production_model_is_loaded_once_and_reused(env):
    env.register()
    first = env.ensure()
    assert env.service.loaded_version == "v1"
    assert env.ensure() is first  # 다시 부를 때 다시 불러오지 않는다


def test_only_the_production_version_is_loaded(env):
    clone_version(env.root, "v1", "v2")
    env.register("v1", Stage.ARCHIVED)
    env.register("v2", Stage.PRODUCTION)
    env.ensure()
    assert env.service.loaded_version == "v2"


def test_new_production_in_db_is_picked_up_without_restart(env):
    clone_version(env.root, "v1", "v2", threshold=0.8)
    env.register("v1", Stage.PRODUCTION)
    env.register("v2", Stage.CANDIDATE, threshold=0.8)
    assert env.predict()[0].model_version == "v1"
    env.set_production("v1", "v2")
    output, threshold = env.predict()
    assert output.model_version == "v2" and threshold == 0.8 and env.service.loaded_version == "v2"


# ---------- 판정 ----------

def test_predict_gives_output_and_the_threshold_of_the_loaded_model(env):
    clone_version(env.root, "v1", "v2", threshold=0.8)  # 설정 파일의 기본값(0.5)이 아니라 모델에 기록된 값을 쓴다 (C5)
    env.register("v1", Stage.ARCHIVED)
    env.register("v2", Stage.PRODUCTION, threshold=0.8)
    output, threshold = env.predict()
    assert threshold == 0.8 and output.model_version == "v2"
    assert output.predicted_label.value == ("NG" if output.score >= 0.8 else "OK")
    assert 0.0 <= output.score <= 1.0 and isinstance(output.latency_ms, int) and 0 <= output.latency_ms <= 5000


def test_predict_matches_the_predictor_used_directly(env):
    from parts.predictor import Predictor

    env.register()
    direct = Predictor(root=env.root)
    direct.load("v1")
    for i in range(3):
        output, _ = env.predict(i)
        assert output.score == direct.predict(env.models.images[i]).score


def test_predict_without_production_is_unavailable(env):
    with pytest.raises(ModelUnavailableError, match=NO_MODEL):
        env.predict()


# ---------- 불러오기 실패: 요청마다 다시 시도, 로그는 한 번, 고치면 복구 ----------

def test_broken_files_are_unavailable_and_retried_every_time(env, caplog):
    env.register()
    env.break_files()
    with caplog.at_level(logging.WARNING, logger="backend"):
        for _ in range(3):
            with pytest.raises(ModelUnavailableError, match=LOAD_FAILED):
                env.ensure()
    assert env.service.loaded_version is None
    assert failure_logs(caplog) == 1  # 같은 실패는 한 번만 기록한다


def test_fixing_the_files_recovers_without_restart_and_a_new_failure_is_logged_again(env, caplog):
    env.register()
    env.break_files()
    with caplog.at_level(logging.WARNING, logger="backend"):
        with pytest.raises(ModelUnavailableError):
            env.ensure()
        env.fix_files()
        assert env.predict()[0].model_version == "v1" and env.service.loaded_version == "v1"  # 재시작 없이 복구
        env.service.reload("v1")  # 정상 reload
        env.break_files()
        with pytest.raises(ModelUnavailableError):
            env.service.reload("v1")
    assert failure_logs(caplog) == 2  # 복구 뒤의 새 실패는 다시 기록한다


def test_different_failures_are_each_logged(env, caplog):
    env.register()
    env.break_files()
    with caplog.at_level(logging.WARNING, logger="backend"):
        with pytest.raises(ModelUnavailableError):
            env.ensure()
        (env.root / "models" / "v1" / "metrics.json").write_text("{}", encoding="utf-8")  # 다른 종류의 실패
        with pytest.raises(ModelUnavailableError):
            env.ensure()
    assert failure_logs(caplog) == 2


def test_missing_model_folder_is_unavailable(env):
    env.register()
    import shutil
    shutil.rmtree(env.root / "models" / "v1")
    with pytest.raises(ModelUnavailableError, match=LOAD_FAILED):
        env.ensure()


def test_failure_message_hides_paths_and_internal_errors(env):
    env.register()
    env.break_files()
    with pytest.raises(ModelUnavailableError) as e:
        env.ensure()
    assert str(e.value) == LOAD_FAILED


def test_failed_switch_does_not_serve_the_old_model_but_keeps_it_loaded(env):
    clone_version(env.root, "v1", "v2")
    env.register("v1", Stage.PRODUCTION)
    env.register("v2", Stage.CANDIDATE)
    env.ensure()
    env.break_files("v2")
    env.set_production("v1", "v2")  # DB 는 v2 가 운영인데 v2 파일이 깨져 있다
    with pytest.raises(ModelUnavailableError, match=LOAD_FAILED):
        env.ensure()
    assert env.service.loaded_version == "v1"  # 불러오기에 실패해도 Predictor 는 기존 모델을 그대로 가진다 (계약 4장)
    env.fix_files("v2")
    assert env.predict()[0].model_version == "v2"


# ---------- reload (4주차 승인이 부를 자리) ----------

def test_reload_replaces_the_loaded_model(env):
    clone_version(env.root, "v1", "v2")
    env.register()
    env.ensure()
    env.service.reload("v2")
    assert env.service.loaded_version == "v2"


@pytest.mark.parametrize("bad", ["v9", "../v1", "V1", "", "v1/.."])
def test_reload_failure_keeps_the_current_model(env, bad):
    env.register()
    env.ensure()
    with pytest.raises(ModelUnavailableError):
        env.service.reload(bad)
    assert env.service.loaded_version == "v1"


def test_reload_to_broken_files_keeps_the_current_model(env):
    clone_version(env.root, "v1", "v2")
    env.break_files("v2")
    env.register()
    env.ensure()
    with pytest.raises(ModelUnavailableError):
        env.service.reload("v2")
    assert env.service.loaded_version == "v1"
    assert env.predict()[0].model_version == "v1"


# ---------- 동시 요청 ----------

def test_concurrent_first_requests_load_one_model(tmp_path, file_engine, trained_models):
    env = Env(tmp_path, make_session_factory(file_engine), trained_models)
    env.register()
    results, errors = [], []

    def run():
        try:
            results.append(env.ensure())
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=run) for _ in range(8)]
    [t.start() for t in threads]
    [t.join(30) for t in threads]
    assert errors == [] and len(results) == 8 and len({id(r) for r in results}) == 1  # 모두 같은 Predictor


def test_predictions_stay_consistent_while_the_model_is_replaced(tmp_path, file_engine, trained_models):
    """판정하는 동안 v1(임계값 0.5) <-> v2(임계값 0.8)로 계속 교체해도, 돌려받은 (버전, 임계값)은 항상 짝이 맞다."""
    env = Env(tmp_path, make_session_factory(file_engine), trained_models)
    clone_version(env.root, "v1", "v2", threshold=0.8)
    env.register("v1", Stage.PRODUCTION)
    env.register("v2", Stage.CANDIDATE, threshold=0.8)
    env.ensure()
    expected = {"v1": 0.5, "v2": 0.8}
    stop, bad, count = threading.Event(), [], []

    def predict_loop():
        while not stop.is_set():
            output, threshold = env.predict()
            count.append(1)
            if threshold != expected[output.model_version]:
                bad.append((output.model_version, threshold))

    def swap_loop():
        for i in range(12):
            env.service.reload("v2" if i % 2 == 0 else "v1")
        stop.set()

    threads = [threading.Thread(target=predict_loop) for _ in range(4)] + [threading.Thread(target=swap_loop)]
    [t.start() for t in threads]
    [t.join(60) for t in threads]
    assert bad == [] and len(count) > 0


# ---------- 서버가 켜질 때 미리 불러오기 ----------

def test_preload_loads_the_production_model(env):
    env.register()
    env.service.preload(env.factory)
    assert env.service.loaded_version == "v1"


def test_preload_never_stops_the_server(env, caplog):
    env.service.preload(env.factory)  # 운영 모델 없음
    assert env.service.loaded_version is None
    env.register()
    env.break_files()
    env.service.preload(env.factory)  # 파일 깨짐
    assert env.service.loaded_version is None
    env.service.ensure_loaded = lambda session: (_ for _ in ()).throw(RuntimeError("뜻밖의 오류"))
    with caplog.at_level(logging.ERROR, logger="backend"):
        env.service.preload(env.factory)  # 그 밖의 오류도 서버 시작을 막지 않는다
    assert "확인하는 중 오류" in caplog.text


def prepared_app(tmp_path, trained_models, *, production=True, broken=False):
    copy_models(trained_models.root, tmp_path)
    if broken:
        (tmp_path / "models" / "v1" / "model.pt").write_bytes(b"not a model")
    app = create_app(make_settings(root=str(tmp_path)))
    if production:
        with session_scope(app.state.session_factory) as s:
            sync_models(s, tmp_path / "models", production="v1")
    return app


def test_lifespan_preloads_so_health_says_loaded_right_after_start(tmp_path, trained_models):
    app = prepared_app(tmp_path, trained_models)
    try:
        assert TestClient(app).get("/health").json()["model"] == {"production_version": "v1", "loaded": False}  # with 없이는 시작 단계가 없다
        with TestClient(app) as client:  # 서버가 켜질 때처럼 lifespan 을 실행한다
            assert client.get("/health").json()["model"] == {"production_version": "v1", "loaded": True}
    finally:
        app.state.engine.dispose()


@pytest.mark.parametrize("production, broken", [(False, False), (True, True)])
def test_server_starts_even_without_a_usable_model(tmp_path, trained_models, production, broken):
    app = prepared_app(tmp_path, trained_models, production=production, broken=broken)
    try:
        with TestClient(app) as client:
            body = client.get("/health").json()
            assert body["status"] == "ok" and body["model"]["loaded"] is False
            assert body["model"]["production_version"] == ("v1" if production else None)
    finally:
        app.state.engine.dispose()
