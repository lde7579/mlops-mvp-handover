"""GET /models, /models/production, /models/{version}: 모델 버전 조회 (모델 파일은 쓰지 않고 DB 의 행만 읽는다)."""
import json
from datetime import datetime

import pytest
from sqlalchemy import text

from backend.inference import NO_MODEL
from tests.backend.helpers import at, bearer, make_metrics

VERSION_KEYS = {"version", "stage", "metrics", "train_data_snapshot", "created_at", "approved_by", "approved_at"}
METRICS_KEYS = {"version", "threshold", "normal", "drift", "train_data_snapshot", "created_at"}
CLASS_KEYS = {"accuracy", "recall", "precision", "f1", "confusion_matrix", "n"}
NOT_FOUND_VERSION = {"error": "not_found", "message": "모델 버전을 찾을 수 없습니다"}


def get(env, path, **params):
    return env.client.get(path, params=params, headers=env.headers)


def versions(response):
    return [i["version"] for i in response.json()["items"]]


# ---------- 인증 ----------

@pytest.mark.parametrize("path", ["/models", "/models/production", "/models/v1"])
def test_requires_login(read_env, path):
    for headers in ({}, {"Authorization": "Bearer garbage"}):
        r = read_env.client.get(path, headers=headers)
        assert r.status_code == 401 and r.json()["error"] == "unauthorized"


@pytest.mark.parametrize("user", ["operator", "admin"])
def test_any_logged_in_role_can_read(read_env, user):
    for path in ("/models", "/models/production", "/models/v1"):
        assert read_env.client.get(path, headers=bearer(read_env.settings, user)).status_code == 200


# ---------- 목록 ----------

def test_list_is_newest_first_with_the_7_contract_fields(read_env):
    r = get(read_env, "/models")
    body = r.json()
    assert r.status_code == 200 and set(body) == {"items", "total"} and body["total"] == 3
    assert versions(r) == ["v3", "v2", "v1"]  # 생성 시각 내림차순
    for item in body["items"]:
        assert set(item) == VERSION_KEYS and set(item["metrics"]) == METRICS_KEYS
        assert set(item["metrics"]["normal"]) == CLASS_KEYS and set(item["metrics"]["drift"]) == CLASS_KEYS


@pytest.mark.parametrize("stage, expected", [
    ("production", ["v2"]), ("archived", ["v1"]), ("candidate", ["v3"]), ("staging", []), ("rejected", []),
])
def test_stage_filter(read_env, stage, expected):
    r = get(read_env, "/models", stage=stage)
    assert r.status_code == 200 and versions(r) == expected and r.json()["total"] == len(expected)


@pytest.mark.parametrize("stage", ["bogus", "Production", "PRODUCTION", ""])
def test_bad_stage_is_400(read_env, stage):
    r = get(read_env, "/models", stage=stage)
    assert r.status_code == 400 and r.json()["error"] == "bad_request" and "stage" in r.json()["message"]


def test_empty_database_gives_an_empty_list(empty_env):
    assert get(empty_env, "/models").json() == {"items": [], "total": 0}


# ---------- 값과 시각 ----------

def test_values_of_the_production_model(read_env):
    v2 = get(read_env, "/models/v2").json()
    assert (v2["version"], v2["stage"], v2["approved_by"]) == ("v2", "production", "admin")
    assert v2["train_data_snapshot"] == {"train": 2000, "pool": 0, "seed": 42}
    assert v2["metrics"]["version"] == "v2" and v2["metrics"]["threshold"] == 0.5
    assert v2["metrics"]["normal"] == {"accuracy": 1.0, "recall": 1.0, "precision": 1.0, "f1": 1.0, "confusion_matrix": [[400, 0], [0, 100]], "n": 500}
    assert v2["metrics"]["drift"]["accuracy"] == 0.9 and v2["metrics"]["drift"]["confusion_matrix"] == [[350, 50], [0, 100]]
    assert v2["metrics"]["train_data_snapshot"] == v2["train_data_snapshot"]


def test_not_yet_approved_models_have_null_approval(read_env):
    for version in ("v1", "v3"):
        item = get(read_env, f"/models/{version}").json()
        assert item["approved_by"] is None and item["approved_at"] is None


def test_all_times_are_utc_z_and_the_same_instant_as_stored(read_env):
    v2 = get(read_env, "/models/v2").json()
    for value in (v2["created_at"], v2["approved_at"], v2["metrics"]["created_at"]):
        assert value.endswith("Z") and datetime.fromisoformat(value).utcoffset().total_seconds() == 0
    assert datetime.fromisoformat(v2["created_at"]) == at(1) == datetime.fromisoformat(v2["metrics"]["created_at"])
    assert datetime.fromisoformat(v2["approved_at"]) == at(5)
    with read_env.app.state.engine.connect() as conn:  # 저장된 metrics.json 내용에는 현지 표기(+09:00)가 그대로 있다
        stored = json.loads(conn.execute(text("SELECT metrics FROM model_version WHERE version='v2'")).scalar())
    assert stored["created_at"].endswith("+09:00") and not v2["metrics"]["created_at"].endswith("+09:00")


def test_only_the_times_are_converted_the_numbers_are_untouched(read_env):
    got = get(read_env, "/models/v3").json()["metrics"]
    expected = make_metrics("v3", created_minutes=2).to_dict()
    for key in ("version", "threshold", "normal", "drift", "train_data_snapshot"):
        assert got[key] == expected[key], key


# ---------- production ----------

def test_production_returns_the_production_model(read_env):
    r = get(read_env, "/models/production")
    assert r.status_code == 200 and r.json()["version"] == "v2" and r.json() == get(read_env, "/models/v2").json()


def test_no_production_is_404_not_503(empty_env):
    r = get(empty_env, "/models/production")
    assert r.status_code == 404 and r.json() == {"error": "not_found", "message": NO_MODEL}  # /predict 의 503 과 문구는 같고 코드가 다르다


# ---------- /models/{version} ----------

def test_one_version_equals_its_list_item(read_env):
    items = {i["version"]: i for i in get(read_env, "/models").json()["items"]}
    for version in ("v1", "v2", "v3"):
        assert get(read_env, f"/models/{version}").json() == items[version]


@pytest.mark.parametrize("name", [
    "v9", "v10", "v0",                       # 형식은 맞지만 없는 버전
    "V1", "v01", "v", "v1a", "v-1", "v1.0",  # 형식이 다르다 (v01 은 형식은 맞지만 없는 이름)
    "v٣", "PRODUCTION", "Production", "production2", "latest",   # 유니코드 숫자, 예약어와 비슷한 이름
    "v1%20", "%20v1", "v1%0A", "v%2E1",      # 공백, 줄바꿈, 점
    "v" * 300, "v" + "1" * 300, "1" * 300,   # 아주 긴 이름
])
def test_bad_or_unknown_version_names_are_404(read_env, name):
    r = read_env.client.get(f"/models/{name}", headers=read_env.headers)
    assert r.status_code == 404 and r.json() == NOT_FOUND_VERSION


def test_production_is_not_treated_as_a_version_name(read_env):
    assert get(read_env, "/models/production").json()["version"] == "v2"  # 경로 순서: production 이 먼저


def test_not_found_on_an_empty_database(empty_env):
    assert get(empty_env, "/models/v1").json() == NOT_FOUND_VERSION


def test_openapi_has_the_three_paths_with_400_or_404_not_422(read_env):
    paths = read_env.client.get("/openapi.json").json()["paths"]
    assert "400" in paths["/models"]["get"]["responses"]
    for path in ("/models", "/models/production", "/models/{version}"):
        assert "422" not in paths[path]["get"]["responses"]
    assert "404" in paths["/models/production"]["get"]["responses"] and "404" in paths["/models/{version}"]["get"]["responses"]
