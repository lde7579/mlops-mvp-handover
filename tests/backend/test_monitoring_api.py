"""GET /monitoring/performance, /monitoring/drift (api-v1.md A24·A25). 메모리 DB, 가짜 시계·가짜 이미지 읽기 함수 (수집기·torch 없음).

판정 행은 helpers.seed_read_only_data 가 넣는 것을 쓴다 (검사 12건: 1~6 정답 NG, 7~12 정답 OK / v1 archived, v2 production, v3 candidate):
  v1 판정 10건 (at(101)~at(110)): TP, TP, FN, TN, FP, TP, TN, FN, TN, TP
  v2 판정 4건 (at(201)~at(204)): TN, FP, TN, TN
  v3 판정 2건 (둘 다 at(300)): TP, FP
읽기만 하는 시험은 모듈 하나의 앱을 함께 쓰고, 행을 바꾸는 시험은 시험마다 새 앱을 만든다.
"""
from datetime import timezone
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.database import session_scope
from backend.main import create_app
from backend.monitoring import ACCURACY_THRESHOLD, CRITERIA_CONFIRMED, RECENT_N, MonitoringService, _version_key
from backend.repository import InspectionRepository, ModelVersionRepository, Stage
from tests.backend.helpers import (
    at,
    bearer,
    insert_predictions,
    make_inspection,
    make_metrics,
    make_settings,
    seed_read_only_data,
    write_png,
)

NOW = at(120)
CRITERIA = {"recent_n": 30, "accuracy_threshold": 0.95, "confirmed": True}


def iso(minutes: int) -> str:
    """at(minutes) 를 UTC 'Z' 표기로 (응답의 시각 표기와 같다)."""
    return at(minutes).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


q = iso  # 쿼리에 넣을 시각도 같은 표기 ('+09:00' 대신 'Z' 를 써서 URL 인코딩이 필요 없다)


def pixels(brightness: float, noise: float) -> np.ndarray:
    """평균 밝기가 brightness, 노이즈 정도(이웃한 가로 픽셀 차이의 평균)가 noise 인 작은 배열."""
    return np.tile([brightness - noise / 2, brightness + noise / 2], (4, 2))


class Reader:
    """이미지 읽기 함수 대신 넣는 가짜: 파일 이름(확장자 뺀 것)으로 (밝기, 노이즈)를 정하고 읽은 횟수를 센다."""

    def __init__(self, table=None):
        self.table = dict(table or {})
        self.reads = []

    def __call__(self, path):
        self.reads.append(path.stem)
        value = self.table.get(path.stem)
        return None if value is None else pixels(*value)


def build(root, *, seed=True, reader=None, **service_args):
    settings = make_settings(root=str(root))
    app = create_app(settings)
    app.state.monitoring = MonitoringService(root, clock=lambda: NOW, reader=reader if reader is not None else Reader(), **service_args)
    if seed:
        with session_scope(app.state.session_factory) as s:
            seed_read_only_data(s)
    return SimpleNamespace(app=app, client=TestClient(app, raise_server_exceptions=False), headers=bearer(settings, "operator"),
                           root=root, reader=app.state.monitoring._reader)


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    e = build(tmp_path_factory.mktemp("monitoring_shared"))
    yield e
    e.app.state.engine.dispose()


@pytest.fixture
def env(tmp_path):
    e = build(tmp_path)
    yield e
    e.app.state.engine.dispose()


@pytest.fixture
def empty(tmp_path):
    e = build(tmp_path, seed=False)
    yield e
    e.app.state.engine.dispose()


def get(e, path, **params):
    return e.client.get(path, params=params, headers=e.headers)


def perf(e, **params):
    return get(e, "/monitoring/performance", **params)


# ---------- 공통 ----------

@pytest.mark.parametrize("path", ["/monitoring/performance", "/monitoring/drift"])
def test_requires_login_and_post_is_405(shared, path):
    r = shared.client.get(path)
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    r = shared.client.post(path, headers=shared.headers)
    assert r.status_code == 405 and "allow" in r.headers


def test_both_paths_are_in_openapi_and_admin_can_read(shared):
    paths = shared.client.get("/openapi.json").json()["paths"]
    assert "/monitoring/performance" in paths and "/monitoring/drift" in paths
    admin = bearer(make_settings(), "admin")
    assert shared.client.get("/monitoring/drift", headers=admin).status_code == 200


# ---------- performance: 형식·구간 ----------

def test_empty_database_gives_empty_points_and_null_recent(empty):
    r = perf(empty)
    assert r.status_code == 200
    assert r.json() == {
        "since": iso(60), "until": iso(120), "bucket_seconds": 30, "criteria": CRITERIA,  # 기본: 지금(가짜 시계)과 그 1시간 전
        "recent": {"model_version": None, "n": 0, "accuracy": None, "recall": None, "confusion_matrix": [[0, 0], [0, 0]]},
        "points": []}


def test_points_are_grouped_per_bucket_and_version_with_exact_values(shared):
    r = perf(shared, since=q(100), until=q(310), bucket_seconds=600)
    assert r.status_code == 200
    body = r.json()
    assert (body["since"], body["until"], body["bucket_seconds"]) == (iso(100), iso(310), 600)
    assert [(p["bucket_start"], p["model_version"], p["n"]) for p in body["points"]] == [
        (iso(100), "v1", 9), (iso(110), "v1", 1), (iso(200), "v2", 4), (iso(300), "v3", 2)]
    by = {(p["bucket_start"], p["model_version"]): p for p in body["points"]}
    first = by[(iso(100), "v1")]  # TP TP FN TN FP TP TN FN TN -> TN 3, FP 1, FN 2, TP 3
    assert first["accuracy"] == pytest.approx(6 / 9) and first["recall"] == pytest.approx(3 / 5)
    assert by[(iso(110), "v1")]["accuracy"] == 1.0 and by[(iso(110), "v1")]["recall"] == 1.0
    v2 = by[(iso(200), "v2")]  # TN FP TN TN: 정답 NG 가 없어서 recall 은 null
    assert v2["accuracy"] == 0.75 and v2["recall"] is None
    assert by[(iso(300), "v3")]["accuracy"] == 0.5 and by[(iso(300), "v3")]["recall"] == 1.0


def test_small_buckets_split_the_same_minutes(shared):
    body = perf(shared, since=q(101), until=q(105), bucket_seconds=60).json()
    assert [(p["bucket_start"], p["n"], p["accuracy"], p["recall"]) for p in body["points"]] == [
        (iso(101), 1, 1.0, 1.0), (iso(102), 1, 1.0, 1.0), (iso(103), 1, 0.0, 0.0), (iso(104), 1, 1.0, None)]


def test_since_is_inclusive_and_until_is_exclusive(shared):
    body = perf(shared, since=q(101), until=q(103), bucket_seconds=600).json()
    assert [(p["model_version"], p["n"]) for p in body["points"]] == [("v1", 2)]  # 101, 102 만 (103 은 빠진다)


def test_default_window_is_one_hour_before_the_fake_now(shared):
    body = perf(shared).json()
    assert (body["since"], body["until"]) == (iso(60), iso(120))
    assert sum(p["n"] for p in body["points"]) == 10  # v1 판정 10건뿐 (v2 는 at(201) 이후)
    assert perf(shared, until=q(200)).json()["since"] == iso(140)  # until 만 주면 그 1시간 전
    assert perf(shared, since=q(110)).json()["until"] == iso(120)  # since 만 주면 지금까지


def test_model_version_filter_limits_points_and_recent(shared):
    body = perf(shared, since=q(0), until=q(400), bucket_seconds=3600, model_version="v2").json()
    assert {p["model_version"] for p in body["points"]} == {"v2"} and body["recent"]["model_version"] == "v2"
    assert perf(shared, model_version="v9").json()["recent"] == {
        "model_version": "v9", "n": 0, "accuracy": None, "recall": None, "confusion_matrix": [[0, 0], [0, 0]]}


# ---------- performance: 최근 N건 (기본은 현재 production) ----------

def test_recent_defaults_to_the_production_version_only(shared):
    recent = perf(shared).json()["recent"]  # production 은 v2: 판정 4건 TN FP TN TN
    assert recent == {"model_version": "v2", "n": 4, "accuracy": 0.75, "recall": None, "confusion_matrix": [[3, 1], [0, 0]]}


def test_recent_for_an_explicit_version_and_window_size(shared):
    v1 = perf(shared, model_version="v1").json()["recent"]
    assert v1["n"] == 10 and v1["confusion_matrix"] == [[3, 1], [2, 4]] and v1["accuracy"] == 0.7 and v1["recall"] == pytest.approx(4 / 6)
    last5 = perf(shared, model_version="v1", recent=5).json()["recent"]  # 최신 5건: at(110)~at(106) = TP TN FN TN TP
    assert last5["confusion_matrix"] == [[2, 0], [1, 2]] and last5["accuracy"] == 0.8 and last5["recall"] == pytest.approx(2 / 3)


def test_criteria_is_the_confirmed_value(shared):
    assert (RECENT_N, ACCURACY_THRESHOLD, CRITERIA_CONFIRMED) == (30, 0.95, True)
    assert perf(shared).json()["criteria"] == CRITERIA
    assert perf(shared, recent=7).json()["criteria"] == {**CRITERIA, "recent_n": 7}  # 쓴 N 을 그대로 알려 준다


# ---------- performance: 오류 ----------

@pytest.mark.parametrize("params", [
    {"since": q(110), "until": q(110)},                 # 같음
    {"since": q(111), "until": q(110)},                 # since 가 더 늦음
    {"since": "2026-10-18T09:00:00"},                   # 시간대 없음
    {"until": "어제"}, {"bucket_seconds": 9}, {"bucket_seconds": 3601}, {"bucket_seconds": "abc"},
    {"recent": 0}, {"recent": 1001}, {"model_version": ""},
])
def test_bad_queries_are_400(shared, params):
    r = perf(shared, **params)
    assert r.status_code == 400 and r.json()["error"] == "bad_request"


def test_since_not_before_until_gets_a_clear_message(shared):
    assert "since" in perf(shared, since=q(5), until=q(5)).json()["message"]


def test_bucket_count_limit_is_720(shared):
    assert perf(shared, since=q(0), until=q(120), bucket_seconds=10).status_code == 200  # 7200초 / 10초 = 정확히 720개
    over = perf(shared, since=q(0), until="2026-10-18T02:00:05Z", bucket_seconds=10)  # 721개
    assert over.status_code == 400 and "720" in over.json()["message"]
    assert perf(shared, since=q(0), until=q(121), bucket_seconds=10).status_code == 400


# ---------- performance: data_split 과 무관 ----------

def test_numbers_do_not_change_when_pool_becomes_train(env):
    before = (perf(env, since=q(100), until=q(310), bucket_seconds=600).json(), perf(env).json(), get(env, "/monitoring/drift").json())
    with session_scope(env.app.state.session_factory) as s:  # 재학습으로 pool 이 train 으로 편입된 상태
        assert InspectionRepository(s).merge_pool_into_train([f"INS-{n:06d}" for n in range(1, 13)]) == 12
        assert InspectionRepository(s).count(data_split="pool") == 0
    after = (perf(env, since=q(100), until=q(310), bucket_seconds=600).json(), perf(env).json(), get(env, "/monitoring/drift").json())
    assert before == after


# ---------- drift: 성능 저하 경고 ----------

def test_warning_without_production_is_on_hold(empty):
    r = get(empty, "/monitoring/drift")
    assert r.status_code == 200
    assert r.json()["performance_warning"] == {"model_version": None, "active": None, "n": 0, "accuracy": None, "recent_n": 30,
                                              "accuracy_threshold": 0.95, "confirmed": True}
    d = r.json()["input_distribution"]
    assert (d["window_n"], d["baseline"], d["current"], d["distance"]) == (0, None, None, None) and isinstance(d["note"], str) and d["note"]


def test_warning_waits_until_n_predictions_exist(shared):
    w = get(shared, "/monitoring/drift").json()["performance_warning"]  # production v2 는 판정 4건뿐 (N=30)
    assert (w["model_version"], w["active"], w["n"], w["accuracy"], w["recent_n"]) == ("v2", None, 4, 0.75, 30)


def test_warning_is_true_below_the_threshold_and_false_above(shared):
    low = get(shared, "/monitoring/drift", recent=4).json()["performance_warning"]  # v2 정확도 0.75 < 0.95
    assert (low["active"], low["n"], low["accuracy"], low["recent_n"]) == (True, 4, 0.75, 4)
    high = get(shared, "/monitoring/drift", recent=2, model_version="v1").json()["performance_warning"]  # v1 최신 2건 TP TN
    assert (high["active"], high["accuracy"], high["model_version"]) == (False, 1.0, "v1")


def _predictions(env, version, wrong, total, first_id):
    """INS-000001(정답 NG)을 total 번 판정한 행: 마지막(가장 최신) wrong 건은 OK(오판), 나머지는 NG(정판)."""
    rows = [dict(id=f"PRD-{first_id + k:06d}", ins="INS-000001", ver=version, label="OK" if k >= total - wrong else "NG", minutes=first_id + k)
            for k in range(total)]
    with session_scope(env.app.state.session_factory) as s:
        insert_predictions(s.connection(), rows)


def _only_inspection(env, versions=("v1",)):
    with session_scope(env.app.state.session_factory) as s:
        models = ModelVersionRepository(s)
        for i, v in enumerate(versions):
            models.register(make_metrics(v, created_minutes=i), stage=Stage.PRODUCTION if i == 0 else Stage.ARCHIVED)
        InspectionRepository(s).add(make_inspection(1))


def _promote(env, version):
    with session_scope(env.app.state.session_factory) as s:
        models = ModelVersionRepository(s)
        models.archive_production()
        models.transition(version, Stage.PRODUCTION, approved_by="admin", approved_at=at(5))


def test_threshold_boundary_exactly_095_is_not_a_warning(empty):
    _only_inspection(empty)
    _predictions(empty, "v1", wrong=1, total=20, first_id=1)  # 정확도 19/20 = 0.95
    w = get(empty, "/monitoring/drift", recent=20).json()["performance_warning"]
    assert (w["accuracy"], w["active"]) == (0.95, False)
    _predictions(empty, "v1", wrong=1, total=1, first_id=21)  # 최신 20건에 오판 2건 -> 0.90
    w = get(empty, "/monitoring/drift", recent=20).json()["performance_warning"]
    assert (w["accuracy"], w["active"]) == (0.9, True)


def test_warning_does_not_linger_after_approval_or_rollback(empty):
    """변경 1: 기본 대상은 현재 production 의 판정뿐이라서, 모델이 바뀌면 이전 모델의 나쁜 판정 때문에 경고가 남지 않는다."""
    _only_inspection(empty, versions=("v1", "v2"))
    _predictions(empty, "v1", wrong=30, total=30, first_id=1)  # v1 최근 30건이 모두 오판
    w = get(empty, "/monitoring/drift").json()["performance_warning"]
    assert (w["model_version"], w["active"], w["n"], w["accuracy"]) == ("v1", True, 30, 0.0)

    _promote(empty, "v2")  # 승인: v2 가 production (아직 판정 없음)
    w = get(empty, "/monitoring/drift").json()["performance_warning"]
    assert (w["model_version"], w["active"], w["n"], w["accuracy"]) == ("v2", None, 0, None)  # 이전 모델 때문에 true 가 남지 않는다
    assert perf(empty).json()["recent"]["model_version"] == "v2" and perf(empty).json()["recent"]["n"] == 0

    _predictions(empty, "v2", wrong=0, total=30, first_id=31)  # v2 가 30건 모두 정판
    w = get(empty, "/monitoring/drift").json()["performance_warning"]
    assert (w["model_version"], w["active"], w["accuracy"]) == ("v2", False, 1.0)

    _promote(empty, "v1")  # 롤백: v1 이 다시 production -> v1 의 나쁜 판정 기록이 다시 기준
    w = get(empty, "/monitoring/drift").json()["performance_warning"]
    assert (w["model_version"], w["active"], w["accuracy"]) == ("v1", True, 0.0)
    # 차트용 points 는 버전별로 나뉘어 둘 다 보인다
    versions = {p["model_version"] for p in perf(empty, since=q(0), until=q(100), bucket_seconds=3600).json()["points"]}
    assert versions == {"v1", "v2"}


# ---------- drift: 입력 분포 ----------

def write_world(root, window_ids=(7, 11, 12), baseline=True):
    for n in window_ids:
        write_png(root, f"data/pool/INS-{n:06d}.png", np.zeros((4, 4), dtype=np.uint8))
    if baseline:
        for name in ("T1", "T2", "T3", "T4"):
            write_png(root, f"data/dataset/train/{name}.png", np.zeros((4, 4), dtype=np.uint8))
        (root / "data" / "dataset" / "train" / "notes.txt").write_text("png 가 아니라서 기준에 쓰이지 않는다")


TABLE = {"T1": (100, 8), "T2": (100, 12), "T3": (110, 8), "T4": (110, 12),   # 기준: 밝기 평균 105 표준편차 5, 노이즈 평균 10 표준편차 2
         "INS-000007": (95, 10), "INS-000011": (105, 14), "INS-000012": (115, 18)}


@pytest.fixture
def dist(tmp_path):
    write_world(tmp_path)
    e = build(tmp_path, reader=Reader(TABLE))
    yield e
    e.app.state.engine.dispose()


def test_distribution_values_baseline_current_and_sigma_distance(dist):
    d = get(dist, "/monitoring/drift").json()["input_distribution"]  # v2 최신 4건의 검사: 11, 12, 11, 7
    assert d["window_n"] == 4
    assert d["baseline"] == {"source": "data/dataset/train", "n": 4, "brightness": {"mean": 105.0, "std": 5.0}, "noise": {"mean": 10.0, "std": 2.0}}
    assert d["current"] == {"brightness": pytest.approx(105.0), "noise": pytest.approx(14.0)}  # (105+115+105+95)/4, (14+18+14+10)/4
    assert d["distance"] == {"brightness_sigma": pytest.approx(0.0), "noise_sigma": pytest.approx(2.0)}


def test_distance_follows_the_window_version(dist):
    d = get(dist, "/monitoring/drift", model_version="v3", recent=2).json()  # v3 판정 2건: 검사 1(NG), 7 -> 이미지 1 은 읽을 수 없어 빠진다
    assert d["performance_warning"]["model_version"] == "v3"
    assert d["input_distribution"]["window_n"] == 1
    assert d["input_distribution"]["current"] == {"brightness": 95.0, "noise": 10.0}
    assert d["input_distribution"]["distance"] == {"brightness_sigma": pytest.approx(-2.0), "noise_sigma": pytest.approx(0.0)}


def test_images_are_read_once_and_baseline_is_computed_once(dist):
    get(dist, "/monitoring/drift")
    first = list(dist.reader.reads)
    assert sorted(first) == ["INS-000007", "INS-000011", "INS-000012", "T1", "T2", "T3", "T4"]  # 같은 검사를 두 번 판정해도 한 번만 읽는다
    for _ in range(3):
        get(dist, "/monitoring/drift")
    assert dist.reader.reads == first  # 폴링을 반복해도 파일을 다시 읽지 않는다


def test_only_new_images_are_read_on_the_next_poll(dist):
    get(dist, "/monitoring/drift")
    before = len(dist.reader.reads)
    write_png(dist.root, "data/pool/INS-000008.png", np.zeros((4, 4), dtype=np.uint8))
    dist.reader.table["INS-000008"] = (100, 10)
    with session_scope(dist.app.state.session_factory) as s:
        insert_predictions(s.connection(), [dict(id="PRD-000099", ins="INS-000008", ver="v2", label="OK", minutes=205)])
    d = get(dist, "/monitoring/drift").json()["input_distribution"]
    assert dist.reader.reads[before:] == ["INS-000008"] and d["window_n"] == 5


def test_unreadable_image_is_skipped_and_retried_later(dist):
    (dist.root / "data/pool/INS-000012.png").unlink()
    assert get(dist, "/monitoring/drift").json()["input_distribution"]["window_n"] == 3  # 11, 11, 7
    write_png(dist.root, "data/pool/INS-000012.png", np.zeros((4, 4), dtype=np.uint8))
    assert get(dist, "/monitoring/drift").json()["input_distribution"]["window_n"] == 4  # 기억하지 않았으므로 다시 읽는다


def test_cache_limit_forgets_the_oldest_values(tmp_path):
    write_world(tmp_path)
    e = build(tmp_path, reader=Reader(TABLE), cache_limit=1)
    try:
        get(e, "/monitoring/drift")
        before = len(e.reader.reads)
        get(e, "/monitoring/drift")
        assert len(e.reader.reads) > before  # 한 개만 기억하므로 일부는 다시 읽는다
        assert get(e, "/monitoring/drift").json()["input_distribution"]["window_n"] == 4
    finally:
        e.app.state.engine.dispose()


def test_without_the_train_folder_baseline_and_distance_are_null(tmp_path):
    write_world(tmp_path, baseline=False)
    e = build(tmp_path, reader=Reader(TABLE))
    try:
        d = get(e, "/monitoring/drift").json()["input_distribution"]
        assert (d["baseline"], d["distance"]) == (None, None)
        assert d["window_n"] == 4 and d["current"]["brightness"] == pytest.approx(105.0)  # 현재 값은 그대로 나온다
    finally:
        e.app.state.engine.dispose()


def test_zero_baseline_std_makes_that_distance_null(tmp_path):
    write_world(tmp_path)
    table = {**TABLE, "T1": (100, 8), "T2": (100, 8), "T3": (100, 8), "T4": (100, 8)}  # 기준이 전부 같다: 표준편차 0
    e = build(tmp_path, reader=Reader(table))
    try:
        d = get(e, "/monitoring/drift").json()["input_distribution"]
        assert d["baseline"]["brightness"] == {"mean": 100.0, "std": 0.0}
        assert d["distance"] == {"brightness_sigma": None, "noise_sigma": None}
    finally:
        e.app.state.engine.dispose()


def test_baseline_sample_size_limits_how_many_train_images_are_read(tmp_path):
    write_world(tmp_path)
    e = build(tmp_path, reader=Reader(TABLE), baseline_sample=2)
    try:
        d = get(e, "/monitoring/drift").json()["input_distribution"]
        assert d["baseline"]["n"] == 2 and sorted(r for r in e.reader.reads if r.startswith("T")) == ["T1", "T4"]  # 균등 간격 (처음과 끝)
    finally:
        e.app.state.engine.dispose()


def test_distribution_is_values_only_no_alert_fields(dist):
    d = get(dist, "/monitoring/drift").json()
    assert set(d) == {"performance_warning", "input_distribution"}
    assert set(d["input_distribution"]) == {"window_n", "baseline", "current", "distance", "note"}
    assert set(d["performance_warning"]) == {"model_version", "active", "n", "accuracy", "recent_n", "accuracy_threshold", "confirmed"}


def test_version_sort_key_orders_numbers_not_text():
    assert sorted(["v10", "v2", "v1", "x"], key=_version_key) == ["v1", "v2", "v10", "x"]
