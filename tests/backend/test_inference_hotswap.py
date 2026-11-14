"""InferenceService.prepare / activate / discard: 먼저 읽어 두고, 커밋 뒤에 객체만 바꾼다 (api-v1.md 4장 A18). 가짜 모델 로더, torch 없음."""
import threading

import pytest
from sqlalchemy import text

from backend.database import session_scope
from backend.inference import LOAD_FAILED, InferenceService, ModelUnavailableError
from backend.repository import ModelVersionRepository, Stage
from tests.backend.helpers import FakeParts, make_metrics, make_settings


class Env:
    def __init__(self, factory):
        self.factory, self.fake = factory, FakeParts()
        self.service = InferenceService(make_settings(), parts_loader=self.fake)
        with session_scope(factory) as s:
            models = ModelVersionRepository(s)
            models.register(make_metrics("v1"), stage=Stage.PRODUCTION)
            models.register(make_metrics("v2", created_minutes=1), stage=Stage.STAGING)

    def switch_db_to_v2(self):
        with session_scope(self.factory) as s:
            s.execute(text("UPDATE model_version SET stage='archived' WHERE version='v1'"))
            s.execute(text("UPDATE model_version SET stage='production' WHERE version='v2'"))

    def ensure(self):
        with session_scope(self.factory) as s:
            return self.service.ensure_loaded(s)

    def predict(self):
        with session_scope(self.factory) as s:
            output, threshold = self.service.predict(s, None)
        return output.model_version, threshold


@pytest.fixture
def env(factory):
    e = Env(factory)
    e.ensure()  # v1 이 올라와 있는 상태에서 시작한다
    e.fake.loads.clear()
    return e


def test_prepare_reads_the_new_model_without_touching_the_running_one(env):
    prepared = env.service.prepare("v2")
    assert prepared.version == "v2" and env.fake.loads == ["v2"]
    assert env.service.loaded_version == "v1" and env.predict() == ("v1", 0.5)


def test_activate_switches_the_model_and_its_threshold_together(env):
    env.switch_db_to_v2()  # 승인: 커밋 뒤에 교체한다
    env.service.activate(env.service.prepare("v2"))
    assert env.service.loaded_version == "v2" and env.predict() == ("v2", 0.7)
    assert env.fake.loads == ["v2"]  # 교체는 파일을 다시 읽지 않는다


def test_activating_before_the_db_commit_would_flip_back_so_the_db_is_the_source_of_truth(env):
    """그래서 교체는 커밋 뒤에만 한다: DB 가 아직 v1 이면 다음 판정이 v1 로 되돌린다 (방식 A 를 쓰지 않는 이유)."""
    env.service.activate(env.service.prepare("v2"))
    assert env.predict() == ("v1", 0.5)


def test_failed_prepare_raises_and_keeps_the_running_model(env):
    env.fake.broken.add("v2")
    with pytest.raises(ModelUnavailableError, match=LOAD_FAILED):
        env.service.prepare("v2")
    assert env.service.loaded_version == "v1" and env.predict() == ("v1", 0.5)
    env.switch_db_to_v2()
    with pytest.raises(ModelUnavailableError):  # 실패한 모델이 몰래 남아 쓰이지 않는다
        env.ensure()


def test_prepare_rejects_a_bad_version_name_like_a_load_failure(env):
    env.fake.on_load = lambda version: (_ for _ in ()).throw(ValueError("v숫자 형식이 아닙니다"))
    with pytest.raises(ModelUnavailableError):
        env.service.prepare("x1")
    assert env.service.loaded_version == "v1"


def test_ensure_loaded_uses_the_prepared_model_after_the_db_switch_without_rereading(env):
    """커밋은 됐는데 activate 가 실행되지 못한 상황: 다음 판정이 읽어 둔 모델을 그대로 쓴다."""
    env.service.prepare("v2")
    env.switch_db_to_v2()  # 커밋됨. activate 는 부르지 않는다
    assert env.service.loaded_version == "v1"
    assert env.predict() == ("v2", 0.7)
    assert env.fake.loads == ["v2"]  # v2 파일은 prepare 때 한 번만 읽었다


def test_ensure_loaded_reads_the_file_when_nothing_was_prepared(env):
    env.switch_db_to_v2()
    assert env.predict() == ("v2", 0.7) and env.fake.loads == ["v2"]


def test_discard_throws_the_prepared_model_away(env):
    prepared = env.service.prepare("v2")
    env.service.discard(prepared)
    env.switch_db_to_v2()
    env.predict()
    assert env.fake.loads == ["v2", "v2"]  # 버렸으니 파일에서 다시 읽는다


def test_a_prepared_model_of_another_version_is_not_used(env):
    env.service.prepare("v2")
    env.ensure()  # DB 는 아직 v1: 읽어 둔 v2 를 쓰지 않는다
    assert env.service.loaded_version == "v1" and env.fake.loads == ["v2"]


def test_prediction_in_flight_finishes_on_the_old_model_and_threshold(env):
    """판정을 시작한 뒤 모델이 바뀌어도, 그 판정은 이전 모델과 이전 임계값으로 끝난다. 다음 판정부터 새 모델이다."""
    env.fake.predict_gate = threading.Event()
    result = []
    worker = threading.Thread(target=lambda: result.append(env.predict()))
    worker.start()
    assert env.fake.predict_started.wait(5)
    env.service.activate(env.service.prepare("v2"))
    env.fake.predict_gate.set()
    worker.join(5)
    assert result == [("v1", 0.5)]
    env.switch_db_to_v2()
    assert env.predict() == ("v2", 0.7)


def test_reload_still_rereads_the_file_even_when_already_loaded(env):
    env.service.reload("v1")
    assert env.fake.loads == ["v1"] and env.service.loaded_version == "v1"
