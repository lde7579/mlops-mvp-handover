"""GET /predictions: 판정 이력 (읽기 전용 -> 세션 fixture read_env 의 시드를 함께 쓴다. 시드 표는 tests/backend/helpers.py)."""
from datetime import datetime

import pytest

from tests.backend.helpers import at, bearer

ITEM_KEYS = {"prediction_id", "inspection_id", "model_version", "predicted_label", "score", "threshold", "latency_ms",
             "predicted_at", "true_label", "correct"}
NEWEST_FIRST = [f"PRD-{n:06d}" for n in range(16, 0, -1)]


def get(env, **params):
    return env.client.get("/predictions", params=params, headers=env.headers)


def ids(response):
    return [i["prediction_id"] for i in response.json()["items"]]


# ---------- 인증 ----------

def test_requires_login(read_env):
    for headers in ({}, {"Authorization": "Bearer garbage"}):
        r = read_env.client.get("/predictions", headers=headers)
        assert r.status_code == 401 and r.json()["error"] == "unauthorized" and r.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("user", ["operator", "admin"])
def test_any_logged_in_role_can_read(read_env, user):
    assert read_env.client.get("/predictions", headers=bearer(read_env.settings, user)).status_code == 200


# ---------- 형식 ----------

def test_list_format_order_and_total(read_env):
    r = get(read_env)
    body = r.json()
    assert r.status_code == 200 and set(body) == {"items", "total"} and body["total"] == 16
    assert all(set(i) == ITEM_KEYS for i in body["items"]) and ids(r) == NEWEST_FIRST


def test_item_values_and_utc_times(read_env):
    items = {i["prediction_id"]: i for i in get(read_env).json()["items"]}
    wrong_one = items["PRD-000016"]  # 검사 7(정답 OK)을 NG 로 판정
    assert (wrong_one["inspection_id"], wrong_one["model_version"], wrong_one["predicted_label"], wrong_one["true_label"],
            wrong_one["correct"]) == ("INS-000007", "v3", "NG", "OK", False)
    assert (wrong_one["score"], wrong_one["threshold"], wrong_one["latency_ms"]) == (0.9, 0.5, 3)
    right_one = items["PRD-000001"]
    assert (right_one["predicted_label"], right_one["true_label"], right_one["correct"]) == ("NG", "NG", True)
    for item in items.values():
        assert item["predicted_at"].endswith("Z") and datetime.fromisoformat(item["predicted_at"]).utcoffset().total_seconds() == 0
    assert datetime.fromisoformat(wrong_one["predicted_at"]) == at(300)


def test_correct_is_always_prediction_equals_truth(read_env):
    assert all(i["correct"] == (i["predicted_label"] == i["true_label"]) for i in get(read_env).json()["items"])


def test_same_time_is_ordered_by_id_descending(read_env):
    assert ids(get(read_env, limit=2)) == ["PRD-000016", "PRD-000015"]


# ---------- 필터 ----------

@pytest.mark.parametrize("params, expected_ids", [
    ({"model_version": "v1"}, [f"PRD-{n:06d}" for n in range(10, 0, -1)]),
    ({"model_version": "v2"}, ["PRD-000014", "PRD-000013", "PRD-000012", "PRD-000011"]),
    ({"model_version": "v3"}, ["PRD-000016", "PRD-000015"]),
    ({"model_version": "v9"}, []),
    ({"inspection_id": "INS-000011"}, ["PRD-000013", "PRD-000011"]),
    ({"inspection_id": "INS-000007"}, ["PRD-000016", "PRD-000014", "PRD-000004"]),
    ({"inspection_id": "INS-999999"}, []),
    ({"wrong": "true"}, ["PRD-000016", "PRD-000012", "PRD-000008", "PRD-000005", "PRD-000003"]),
    ({"wrong": "false"}, [f"PRD-{n:06d}" for n in (15, 14, 13, 11, 10, 9, 7, 6, 4, 2, 1)]),
    ({"model_version": "v1", "wrong": "true"}, ["PRD-000008", "PRD-000005", "PRD-000003"]),
    ({"inspection_id": "INS-000007", "wrong": "true"}, ["PRD-000016"]),
    ({"model_version": "v2", "inspection_id": "INS-000011"}, ["PRD-000013", "PRD-000011"]),
])
def test_filters(read_env, params, expected_ids):
    r = get(read_env, **params)
    assert r.status_code == 200 and ids(r) == expected_ids and r.json()["total"] == len(expected_ids)


def test_empty_result_has_the_same_shape(read_env):
    assert get(read_env, model_version="v9").json() == {"items": [], "total": 0}


# ---------- 페이지 ----------

def test_paging_and_total_ignore_limit_and_offset(read_env):
    r = get(read_env, limit=5)
    assert ids(r) == NEWEST_FIRST[:5] and r.json()["total"] == 16
    r = get(read_env, limit=5, offset=14)
    assert ids(r) == ["PRD-000002", "PRD-000001"] and r.json()["total"] == 16
    assert get(read_env, offset=16).json() == {"items": [], "total": 16}
    assert len(get(read_env, limit=1).json()["items"]) == 1 and get(read_env, limit=1000).status_code == 200


def test_total_counts_only_the_filtered_rows(read_env):
    r = get(read_env, model_version="v1", limit=2, offset=1)
    assert ids(r) == ["PRD-000009", "PRD-000008"] and r.json()["total"] == 10


# ---------- 잘못된 입력 ----------

@pytest.mark.parametrize("params", [
    {"limit": 0}, {"limit": 1001}, {"limit": -1}, {"limit": "abc"}, {"limit": ""},
    {"offset": -1}, {"offset": "abc"}, {"offset": "1.5"},
    {"wrong": "True"}, {"wrong": "TRUE"}, {"wrong": "1"}, {"wrong": "yes"}, {"wrong": "0"}, {"wrong": "t"}, {"wrong": ""},
    {"model_version": ""}, {"model_version": "v" * 65}, {"inspection_id": ""}, {"inspection_id": "I" * 65},
])
def test_bad_input_is_400_not_422(read_env, params):
    r = get(read_env, **params)
    assert r.status_code == 400 and r.json()["error"] == "bad_request" and set(r.json()) == {"error", "message"}
    assert next(iter(params)) in r.json()["message"]


def test_wrong_message_says_the_value_is_not_allowed(read_env):
    assert get(read_env, wrong="yes").json()["message"] == "wrong: 허용된 값이 아닙니다"


# ---------- 빈 DB, 문서 ----------

def test_empty_database(empty_env):
    assert get(empty_env).json() == {"items": [], "total": 0}


def test_openapi_has_400_but_not_422(read_env):
    responses = read_env.client.get("/openapi.json").json()["paths"]["/predictions"]["get"]["responses"]
    assert "400" in responses and "401" in responses and "422" not in responses
