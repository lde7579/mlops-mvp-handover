"""GET /predictions/stats: 판정 통계. 기대값은 tests/backend/helpers.py 의 시드 표에서 손으로 계산한 값이다."""
import pytest

from tests.backend.helpers import bearer


def stats(env, **params):
    return env.client.get("/predictions/stats", params=params, headers=env.headers)


def cm(body):
    return body["recent"]["confusion_matrix"]


# ---------- 인증 ----------

def test_requires_login(read_env):
    for headers in ({}, {"Authorization": "Bearer garbage"}):
        r = read_env.client.get("/predictions/stats", headers=headers)
        assert r.status_code == 401 and r.json()["error"] == "unauthorized"


@pytest.mark.parametrize("user", ["operator", "admin"])
def test_any_logged_in_role_can_read(read_env, user):
    assert read_env.client.get("/predictions/stats", headers=bearer(read_env.settings, user)).status_code == 200


# ---------- 형식과 손으로 계산한 값 ----------

def test_format(read_env):
    body = stats(read_env).json()
    assert set(body) == {"total", "ng_count", "ng_rate", "recent"}
    assert set(body["recent"]) == {"n", "accuracy", "recall", "confusion_matrix"}


def test_all_versions_default_recent_100(read_env):
    body = stats(read_env).json()  # 16건, NG 로 판정 8건, 최근 100건 = 전체 16건
    assert (body["total"], body["ng_count"], body["ng_rate"]) == (16, 8, 0.5)
    assert body["recent"]["n"] == 16 and cm(body) == [[6, 3], [2, 5]]
    assert body["recent"]["accuracy"] == pytest.approx(11 / 16) and body["recent"]["recall"] == pytest.approx(5 / 7)


def test_v1_only(read_env):
    body = stats(read_env, model_version="v1").json()  # TN3 FP1 FN2 TP4
    assert (body["total"], body["ng_count"], body["ng_rate"]) == (10, 5, 0.5)
    assert body["recent"]["n"] == 10 and cm(body) == [[3, 1], [2, 4]]
    assert body["recent"]["accuracy"] == pytest.approx(0.7) and body["recent"]["recall"] == pytest.approx(4 / 6)


@pytest.mark.parametrize("recent, matrix, accuracy, recall", [
    (1, [[0, 0], [0, 1]], 1.0, 1.0),
    (3, [[1, 0], [1, 1]], 2 / 3, 1 / 2),
    (5, [[2, 0], [1, 2]], 4 / 5, 2 / 3),
    (10, [[3, 1], [2, 4]], 7 / 10, 4 / 6),
    (500, [[3, 1], [2, 4]], 7 / 10, 4 / 6),
])
def test_recent_window_counts_only_the_newest_n_of_v1(read_env, recent, matrix, accuracy, recall):
    body = stats(read_env, model_version="v1", recent=recent).json()
    assert cm(body) == matrix and body["recent"]["n"] == min(recent, 10)
    assert body["recent"]["accuracy"] == pytest.approx(accuracy) and body["recent"]["recall"] == pytest.approx(recall)
    assert (body["total"], body["ng_count"]) == (10, 5)  # total·ng_count 는 recent 와 무관


def test_recent_5_across_all_versions(read_env):
    body = stats(read_env, recent=5).json()  # PRD-16 FP, 15 TP, 14 TN, 13 TN, 12 FP
    assert cm(body) == [[2, 2], [0, 1]] and body["recent"]["accuracy"] == pytest.approx(0.6) and body["recent"]["recall"] == 1.0
    assert body["total"] == 16


def test_recall_is_null_when_the_window_has_no_actual_ng(read_env):
    body = stats(read_env, model_version="v2").json()  # 정답이 모두 OK
    assert (body["total"], body["ng_count"], body["ng_rate"]) == (4, 1, 0.25)
    assert cm(body) == [[3, 1], [0, 0]] and body["recent"]["accuracy"] == 0.75 and body["recent"]["recall"] is None


def test_same_time_predictions_use_id_order(read_env):
    one = stats(read_env, model_version="v3", recent=1).json()  # PRD-16(FP)이 PRD-15(TP)보다 최신
    assert cm(one) == [[0, 1], [0, 0]] and one["recent"]["accuracy"] == 0.0 and one["recent"]["recall"] is None
    two = stats(read_env, model_version="v3", recent=2).json()
    assert cm(two) == [[0, 1], [0, 1]] and two["recent"]["accuracy"] == 0.5 and two["recent"]["recall"] == 1.0


def test_a_prediction_repeated_on_one_inspection_is_counted_each_time(read_env):
    assert stats(read_env, model_version="v2").json()["total"] == 4  # 검사 11 을 두 번 판정한 것이 각각 센다


@pytest.mark.parametrize("recent", [1, 3, 5, 10, 16])
def test_matches_the_top_of_the_prediction_list(read_env, recent):
    items = read_env.client.get("/predictions", params={"limit": recent}, headers=read_env.headers).json()["items"]
    manual = [[0, 0], [0, 0]]
    for i in items:
        manual[1 if i["true_label"] == "NG" else 0][1 if i["predicted_label"] == "NG" else 0] += 1  # [[TN,FP],[FN,TP]]
    assert cm(stats(read_env, recent=recent).json()) == manual


# ---------- 0건 ----------

def test_unknown_version_gives_the_empty_form(read_env):
    assert stats(read_env, model_version="v9").json() == {
        "total": 0, "ng_count": 0, "ng_rate": None,
        "recent": {"n": 0, "accuracy": None, "recall": None, "confusion_matrix": [[0, 0], [0, 0]]}}


def test_empty_database_gives_the_empty_form(empty_env):
    r = stats(empty_env)
    assert r.status_code == 200 and r.json()["recent"] == {"n": 0, "accuracy": None, "recall": None, "confusion_matrix": [[0, 0], [0, 0]]}
    assert (r.json()["total"], r.json()["ng_count"], r.json()["ng_rate"]) == (0, 0, None)


# ---------- 잘못된 입력 ----------

@pytest.mark.parametrize("params", [
    {"recent": 0}, {"recent": 1001}, {"recent": -1}, {"recent": "abc"}, {"recent": ""}, {"recent": "2.5"},
    {"model_version": ""}, {"model_version": "v" * 65},
])
def test_bad_input_is_400(read_env, params):
    r = stats(read_env, **params)
    assert r.status_code == 400 and r.json()["error"] == "bad_request" and next(iter(params)) in r.json()["message"]


def test_recent_boundaries_are_accepted(read_env):
    assert stats(read_env, recent=1).status_code == 200 and stats(read_env, recent=1000).status_code == 200


def test_openapi_has_400_but_not_422(read_env):
    responses = read_env.client.get("/openapi.json").json()["paths"]["/predictions/stats"]["get"]["responses"]
    assert "400" in responses and "422" not in responses
