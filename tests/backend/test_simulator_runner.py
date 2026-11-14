"""자동 생성 러너 (backend/simulator_runner.py, api-v1.md A23·A26). 가짜 생성·판정·시계·대기를 넣어 DB·수집기·torch 없이 시험한다.

sleep 을 쓰지 않는다: 가짜 wait 가 정해진 바퀴 수만큼 바로 돌려보낸 뒤 "대기 중" 이벤트를 켜고 멈춤 신호를 기다린다.
시험은 그 이벤트를 기다렸다가 건수를 확인한다 (이벤트가 안 오면 5초 뒤 실패).
"""
import threading
from datetime import datetime, timezone

import pytest

from backend.inference import ModelUnavailableError
from backend.simulator_runner import MAX_GENERATE_FAILURES, SimulatorBusyError, SimulatorRunner
from tests.backend.helpers import make_inspection

NOW = datetime(2026, 10, 25, 3, 0, 0, tzinfo=timezone.utc)
WAIT_LIMIT = 5.0


class Fakes:
    """cycles 바퀴를 돈 뒤 멈춤 신호를 기다리며 parked 를 켠다. generate 는 호출마다 호출 기록을 남긴다."""

    def __init__(self, cycles=3, generate_errors=(), judge_error=None):
        self.cycles = cycles
        self.calls = []                      # (scenario, label, drift_type)
        self.waits = []                      # 대기 요청 초
        self.judged = []
        self.parked = threading.Event()
        self.generate_errors = list(generate_errors)  # 호출 순서대로 True 면 그 호출은 실패
        self.judge_error = judge_error
        self.on_generate = None              # 호출 번호(0부터)를 받는 함수: 생성 도중에 일어나는 일을 시험한다
        self.gate = None                     # 설정하면 generate 가 이 Event 가 켜질 때까지 기다린다

    def generate(self, scenario, label, drift_type):
        n = len(self.calls)
        self.calls.append((scenario, label, drift_type))
        if self.on_generate is not None:
            self.on_generate(n)
        if self.gate is not None:
            assert self.gate.wait(WAIT_LIMIT)
        if n < len(self.generate_errors) and self.generate_errors[n]:
            raise OSError("디스크 오류")
        return make_inspection(n + 1)

    def judge(self, inspection):
        if self.judge_error is not None:
            raise self.judge_error
        self.judged.append(inspection.inspection_id)

    def wait(self, stop, seconds):
        self.waits.append(seconds)
        if len(self.waits) < self.cycles:
            return False
        self.parked.set()
        return stop.wait(WAIT_LIMIT)

    def runner(self, **kw):
        return SimulatorRunner(self.generate, self.judge, clock=lambda: NOW, wait=self.wait, **kw)


@pytest.fixture
def made():
    """만든 러너를 모아 두었다가 시험이 끝나면 모두 멈춘다 (스레드가 남지 않게)."""
    runners = []

    def make(fakes, **kw):
        r = fakes.runner(**kw)
        runners.append(r)
        return r

    yield make
    for r in runners:
        r.shutdown()
        if r.thread is not None:
            assert not r.thread.is_alive(), "시험이 끝났는데 자동 생성 스레드가 남아 있습니다"


def run(runner, fakes, **start_args):
    runner.start(**start_args)
    assert fakes.parked.wait(WAIT_LIMIT), "러너가 정해진 바퀴를 돌지 못했습니다"


def test_start_returns_status_with_settings_and_clock(made):
    f = Fakes()
    status = made(f).start(0.35, 1.5)
    assert (status.running, status.interval_seconds, status.defect_rate, status.started_at) == (True, 1.5, 0.35, NOW)
    assert (status.generated, status.judged, status.judge_failures, status.last_error) == (0, 0, 0, None)


def test_each_cycle_generates_then_judges_and_passes_interval(made):
    f = Fakes(cycles=3)
    r = made(f)
    run(r, f, defect_rate=0.2, interval_seconds=2.5)
    s = r.status()
    assert (s.generated, s.judged, s.judge_failures) == (3, 3, 0)
    assert f.judged == ["INS-000001", "INS-000002", "INS-000003"]
    assert f.waits == [2.5, 2.5, 2.5]


@pytest.mark.parametrize("rate,expected", [(1.0, "NG"), (0.0, "OK")])
def test_defect_rate_extremes_decide_every_label(made, rate, expected):
    f = Fakes(cycles=6)
    run(made(f), f, defect_rate=rate)
    assert [label for _, label, _ in f.calls] == [expected] * 6


def test_label_sequence_is_reproducible_and_follows_rate(made):
    first, second = Fakes(cycles=40), Fakes(cycles=40)
    run(made(first), first, defect_rate=0.5)
    run(made(second), second, defect_rate=0.5)
    labels = [label for _, label, _ in first.calls]
    assert labels == [label for _, label, _ in second.calls]  # 시드(LABEL_SEED)가 같으면 같은 순서
    assert 5 < labels.count("NG") < 35  # 0.5 근처 (전부 한쪽이 아님)


def test_normal_when_drift_is_off(made):
    f = Fakes(cycles=2)
    run(made(f), f)
    assert [(s, d) for s, _, d in f.calls] == [("normal", None)] * 2


def test_drift_on_before_start_makes_drift_inspections_of_that_type(made):
    f = Fakes(cycles=2)
    r = made(f)
    r.set_drift(True, "noise")
    run(r, f)
    assert [(s, d) for s, _, d in f.calls] == [("drift", "noise")] * 2


def test_drift_toggled_while_running_applies_from_next_cycle(made):
    f = Fakes(cycles=5)
    r = made(f)
    f.on_generate = lambda n: {1: lambda: r.set_drift(True, "dark"), 3: lambda: r.set_drift(False)}.get(n, lambda: None)()
    run(r, f)
    # 호출 1 번째 안에서 켰으므로 그 호출은 이미 normal 로 정해졌고, 다음 바퀴(2)부터 dark. 3 번째 안에서 껐으므로 4 번째부터 normal
    assert [(s, d) for s, _, d in f.calls] == [("normal", None), ("normal", None), ("drift", "dark"), ("drift", "dark"), ("normal", None)]


def test_set_drift_works_while_stopped_and_off_clears_the_type(made):
    r = made(Fakes())
    on = r.set_drift(True, "shift")
    assert (on.running, on.drift_enabled, on.drift_type) == (False, True, "shift")
    off = r.set_drift(False)
    assert (off.drift_enabled, off.drift_type) == (False, None)


def test_judge_failure_skips_judging_but_keeps_generating(made):
    f = Fakes(cycles=4, judge_error=ModelUnavailableError("운영 중인 모델이 없습니다"))
    r = made(f)
    run(r, f)
    s = r.status()
    assert (s.running, s.generated, s.judged, s.judge_failures) == (True, 4, 0, 4)
    assert s.last_error == "판정 건너뜀: 운영 중인 모델이 없습니다"


def test_generate_error_is_recorded_and_next_cycle_still_runs(made):
    f = Fakes(cycles=3, generate_errors=[True, False, False])
    r = made(f)
    run(r, f)
    s = r.status()
    assert (s.running, s.generated, s.judged) == (True, 2, 2)  # 실패한 바퀴는 세지 않는다
    assert s.last_error == "생성 실패: 디스크 오류"


def test_stops_itself_after_consecutive_generate_failures(made):
    f = Fakes(cycles=100, generate_errors=[True] * 100)
    r = made(f)
    r.start()
    r.thread.join(WAIT_LIMIT)
    s = r.status()
    assert not r.thread.is_alive() and s.running is False
    assert len(f.calls) == MAX_GENERATE_FAILURES
    assert s.last_error == "생성 실패: 디스크 오류"  # 멈춘 뒤에도 남는다
    assert (s.interval_seconds, s.defect_rate, s.started_at) == (None, None, None)


def test_a_success_resets_the_failure_streak(made):
    pattern = [True] * (MAX_GENERATE_FAILURES - 1) + [False] + [True] * (MAX_GENERATE_FAILURES - 1) + [False]
    f = Fakes(cycles=len(pattern), generate_errors=pattern)
    r = made(f)
    run(r, f)
    assert r.status().running is True and len(f.calls) == len(pattern)


def test_judge_failures_never_stop_the_runner(made):
    f = Fakes(cycles=MAX_GENERATE_FAILURES * 3, judge_error=RuntimeError("추론 오류"))
    r = made(f)
    run(r, f)
    s = r.status()
    assert s.running is True and s.judge_failures == MAX_GENERATE_FAILURES * 3


def test_start_while_running_raises_busy_and_keeps_running(made):
    f = Fakes()
    r = made(f)
    run(r, f, defect_rate=0.3, interval_seconds=1.0)
    with pytest.raises(SimulatorBusyError, match="이미 실행 중"):
        r.start(0.9, 3.0)
    s = r.status()
    assert (s.running, s.interval_seconds, s.defect_rate) == (True, 1.0, 0.3)  # 첫 설정 그대로


def test_stop_ends_the_thread_and_clears_run_settings(made):
    f = Fakes()
    r = made(f)
    run(r, f)
    s = r.stop()
    assert not r.thread.is_alive()
    assert (s.running, s.interval_seconds, s.defect_rate, s.started_at) == (False, None, None, None)
    assert s.generated == 3  # 멈춘 뒤에도 건수는 남는다


def test_stop_is_idempotent_even_when_never_started(made):
    r = made(Fakes())
    assert r.stop().running is False
    assert r.stop().running is False
    assert r.thread is None


def test_restart_resets_counters_error_and_label_sequence(made):
    f = Fakes(cycles=3, generate_errors=[True])
    r = made(f)
    run(r, f, defect_rate=0.5)
    first_labels = [label for _, label, _ in f.calls]
    r.stop()
    f.calls.clear(), f.waits.clear(), f.parked.clear()
    f.generate_errors = []
    run(r, f, defect_rate=0.5)
    s = r.status()
    assert (s.generated, s.last_error) == (3, None)
    assert [label for _, label, _ in f.calls] == first_labels  # 시작할 때마다 시드(LABEL_SEED)에서 다시 시작


def test_shutdown_stops_a_running_thread(made):
    f = Fakes()
    r = made(f)
    run(r, f)
    r.shutdown()
    assert not r.thread.is_alive() and r.status().running is False


def test_shutdown_never_raises(made):
    r = made(Fakes())
    r.stop = lambda: (_ for _ in ()).throw(RuntimeError("멈추지 못함"))
    r.shutdown()  # 서버 종료를 막지 않는다


def test_stop_does_not_hang_when_a_cycle_is_stuck(made):
    f = Fakes(cycles=1)
    f.gate = threading.Event()
    r = made(f, join_timeout=0.05)
    r.start()
    s = r.stop()  # 생성이 끝나지 않아도 join_timeout 뒤에 돌아온다
    assert s.running is True
    f.gate.set()  # 풀어 주면 스레드는 멈춤 신호를 보고 끝난다
    r.thread.join(WAIT_LIMIT)
    assert not r.thread.is_alive() and r.status().running is False


# ---------- 판정 (backend/judging.py): /predict 와 같은 결과 ----------
import numpy as np
from fastapi.testclient import TestClient

from backend.database import session_scope
from backend.generation import GenerationService
from backend.inference import InferenceService
from backend.judging import JudgeError, judge_inspection
from backend.main import create_app
from backend.repository import InspectionRepository, ModelVersionRepository, PredictionRepository, Stage
from tests.backend.helpers import FakeParts, at, bearer, make_metrics, make_settings, write_png


@pytest.fixture
def judge_world(tmp_path):
    """앱 + 가짜 모델(v1 임계값 0.5 / v2 0.7) + 검사 1건과 그 이미지. v1 이 production, v2 는 archived."""
    settings = make_settings(root=str(tmp_path))
    app = create_app(settings)
    fake = FakeParts()
    inference = InferenceService(settings, parts_loader=fake)
    app.state.inference = inference
    write_png(tmp_path, "data/pool/INS-000001.png", np.random.default_rng(0).integers(0, 256, (128, 128), dtype=np.uint8))
    with session_scope(app.state.session_factory) as s:
        models = ModelVersionRepository(s)
        models.register(make_metrics("v1"), stage=Stage.PRODUCTION)
        models.register(make_metrics("v2", created_minutes=1), stage=Stage.ARCHIVED)
        InspectionRepository(s).add(make_inspection(1))
    yield type("World", (), dict(app=app, root=tmp_path, inference=inference, client=TestClient(app, raise_server_exceptions=False),
                                 op=bearer(settings, "operator")))
    app.state.engine.dispose()


def _judge(world, n=1):
    with session_scope(world.app.state.session_factory) as s:
        return judge_inspection(s, world.inference, world.root, InspectionRepository(s).get(f"INS-{n:06d}"))


def test_judging_gives_the_same_result_as_predict_for_v1_and_v2(judge_world):
    w = judge_world
    for version, threshold in (("v1", 0.5), ("v2", 0.7)):
        if version == "v2":  # 롤백처럼 production 을 v2 로 바꾼다 (추론부는 다음 판정 때 DB 기준으로 새 모델을 불러온다)
            with session_scope(w.app.state.session_factory) as s:
                models = ModelVersionRepository(s)
                models.archive_production()
                models.transition("v2", Stage.PRODUCTION, approved_by="admin", approved_at=at(5))
        api = w.client.post("/predict", json={"inspection_id": "INS-000001"}, headers=w.op)
        assert api.status_code == 201
        judged = _judge(w)
        for field in ("model_version", "predicted_label", "score", "threshold"):
            expected = api.json()[field]
            actual = getattr(judged, field)
            assert (actual.value if hasattr(actual, "value") else actual) == expected, field
        assert (judged.model_version, judged.threshold) == (version, threshold)
    with session_scope(w.app.state.session_factory) as s:
        assert PredictionRepository(s).count() == 4  # 판정마다 한 건씩 저장된다


def test_judging_without_production_model_raises_model_unavailable(judge_world):
    from backend.inference import ModelUnavailableError

    with session_scope(judge_world.app.state.session_factory) as s:
        ModelVersionRepository(s).archive_production()
    with pytest.raises(ModelUnavailableError, match="운영 중인 모델이 없습니다"):
        _judge(judge_world)
    with session_scope(judge_world.app.state.session_factory) as s:
        assert PredictionRepository(s).count() == 0


@pytest.mark.parametrize("bad", ["missing", "wrong_size"])
def test_judging_with_unreadable_image_raises_judge_error_and_saves_nothing(judge_world, bad):
    w = judge_world
    path = w.root / "data/pool/INS-000001.png"
    if bad == "missing":
        path.unlink()
    else:
        write_png(w.root, "data/pool/INS-000001.png", np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(JudgeError):
        _judge(w)
    with session_scope(w.app.state.session_factory) as s:
        assert PredictionRepository(s).count() == 0


# ---------- 생성 옵션 (backend/generation.py): label, drift_type 은 수집기 공개 옵션으로 그대로 넘어간다 ----------

@pytest.fixture
def generation(tmp_path):
    return GenerationService(make_settings(root=str(tmp_path)))


@pytest.mark.parametrize("label", ["OK", "NG"])
def test_generate_label_option_decides_true_label(generation, session, label):
    inspection = generation.generate(session, "normal", label=label)
    assert (inspection.true_label.value, inspection.scenario.value, inspection.data_split.value) == (label, "normal", "pool")
    assert bool(inspection.missing_parts) == (label == "NG")


def test_generate_drift_type_option_is_used_and_stays_in_pool(generation, session):
    inspection = generation.generate(session, "drift", drift_type="noise", label="NG")
    assert (inspection.scenario.value, inspection.drift_type, inspection.data_split.value) == ("drift", "noise", "pool")
