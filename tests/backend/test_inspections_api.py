"""GET /inspections, /inspections/{id}, /inspections/{id}/image (읽기 전용이라 앱 하나를 모듈 안에서 함께 쓴다)."""
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from backend.database import session_scope
from backend.main import create_app
from backend.repository import InspectionRepository, ModelVersionRepository, PredictionRepository
from tests.backend.helpers import (
    at, bearer, make_dir_link, make_inspection, make_metrics, make_output, make_settings, png_bytes, raw_inspection, write_file,
)

ITEM_KEYS = {"inspection_id", "captured_at", "image_path", "true_label", "missing_parts", "scenario", "data_split", "drift_type"}
PREDICTION_KEYS = {"prediction_id", "inspection_id", "model_version", "predicted_label", "score", "threshold", "latency_ms", "predicted_at"}
NOT_FOUND_IMAGE = {"error": "not_found", "message": "이미지를 찾을 수 없습니다"}


def split_of(n: int) -> str:
    return "pool" if n <= 6 else "train" if n <= 9 else "test_drift"


class Env:
    def __init__(self, root):
        self.root = root
        self.settings = make_settings(root=str(root))
        self.app = create_app(self.settings)
        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.headers = bearer(self.settings, "operator")

    def get(self, path, **params):
        return self.client.get(path, params=params, headers=self.headers)

    def scope(self):
        return session_scope(self.app.state.session_factory)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """검사 12건: 1~6 pool, 7~9 train, 10~12 test_drift / 홀수 normal·OK, 짝수 drift(dark)·NG / 2번 검사에 판정 3건."""
    e = Env(tmp_path_factory.mktemp("inspections_api"))
    with e.scope() as s:
        ModelVersionRepository(s).register(make_metrics("v1"))
        repo = InspectionRepository(s)
        for n in range(1, 13):
            even = n % 2 == 0
            rel = f"data/{split_of(n)}/INS-{n:06d}.png"
            repo.add(make_inspection(n, image_path=rel, data_split=split_of(n), scenario="drift" if even else "normal",
                                     drift_type="dark" if even else None, true_label="NG" if even else "OK",
                                     missing_parts=[2] if even else []))
            write_file(e.root, rel, png_bytes(n * 10))
        s.flush()
        for k, label in enumerate(("NG", "OK", "NG")):  # 판정 시각이 오래된 것부터 PRD-000001
            PredictionRepository(s).add(make_output(label, score=0.9 if label == "NG" else 0.1), inspection_id="INS-000002",
                                        threshold=0.5, predicted_at=at(100 + k))
    yield e
    e.app.state.engine.dispose()


def ids(response):
    return [i["inspection_id"] for i in response.json()["items"]]


# ---------- 인증 ----------

@pytest.mark.parametrize("path", ["/inspections", "/inspections/INS-000001", "/inspections/INS-000001/image"])
def test_requires_login(env, path):
    for headers in ({}, {"Authorization": "Bearer garbage"}):
        r = env.client.get(path, headers=headers)
        assert r.status_code == 401 and r.json()["error"] == "unauthorized"
        assert r.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("user", ["operator", "admin"])
def test_any_logged_in_role_can_read(env, user):
    headers = bearer(env.settings, user)
    for path in ("/inspections", "/inspections/INS-000001", "/inspections/INS-000001/image"):
        assert env.client.get(path, headers=headers).status_code == 200


# ---------- 목록 ----------

def test_list_format_and_order(env):
    r = env.get("/inspections")
    body = r.json()
    assert r.status_code == 200 and set(body) == {"items", "total"} and body["total"] == 12 and len(body["items"]) == 12
    assert all(set(i) == ITEM_KEYS for i in body["items"])
    assert ids(r) == [f"INS-{n:06d}" for n in range(12, 0, -1)]  # 최신 먼저


def test_list_item_values(env):
    item = {i["inspection_id"]: i for i in env.get("/inspections").json()["items"]}["INS-000002"]
    assert item["true_label"] == "NG" and item["missing_parts"] == [2] and item["scenario"] == "drift"
    assert item["drift_type"] == "dark" and item["data_split"] == "pool" and item["image_path"] == "data/pool/INS-000002.png"
    assert datetime.fromisoformat(item["captured_at"]).utcoffset() is not None
    assert datetime.fromisoformat(item["captured_at"]) == at(2)
    normal = {i["inspection_id"]: i for i in env.get("/inspections").json()["items"]}["INS-000001"]
    assert normal["drift_type"] is None and normal["missing_parts"] == [] and normal["true_label"] == "OK"


@pytest.mark.parametrize("params, expected_total", [
    ({"scenario": "normal"}, 6), ({"scenario": "drift"}, 6),
    ({"data_split": "pool"}, 6), ({"data_split": "train"}, 3), ({"data_split": "test_drift"}, 3), ({"data_split": "test_normal"}, 0),
    ({"scenario": "drift", "data_split": "train"}, 1),  # 8번
    ({"scenario": "normal", "data_split": "test_drift"}, 1),  # 11번
])
def test_filters(env, params, expected_total):
    r = env.get("/inspections", **params)
    assert r.json()["total"] == expected_total == len(r.json()["items"])


def test_time_range_start_is_included_and_end_is_excluded(env):
    assert ids(env.get("/inspections", since=at(5).isoformat())) == [f"INS-{n:06d}" for n in range(12, 4, -1)]  # 5번 포함
    assert ids(env.get("/inspections", until=at(5).isoformat())) == [f"INS-{n:06d}" for n in range(4, 0, -1)]  # 5번 미포함
    assert ids(env.get("/inspections", since=at(3).isoformat(), until=at(6).isoformat())) == ["INS-000005", "INS-000004", "INS-000003"]


def test_time_range_accepts_other_offsets_and_z(env):
    assert env.get("/inspections", since="2026-10-18T00:05:00Z").json()["total"] == 8  # 09:05 KST = 00:05 UTC
    assert env.get("/inspections", since="2026-10-18T00:05:00+00:00").json()["total"] == 8
    assert env.get("/inspections", since="2026-10-17T19:05:00-05:00").json()["total"] == 8


def test_plus_sign_must_be_encoded_in_the_url(env):
    r = env.client.get("/inspections?since=2026-10-18T09:05:00+09:00", headers=env.headers)  # '+' 가 공백이 된다
    assert r.status_code == 400 and r.json()["error"] == "bad_request"
    assert env.client.get("/inspections?since=2026-10-18T09:05:00%2B09:00", headers=env.headers).status_code == 200


def test_empty_range_with_same_since_and_until_is_allowed(env):
    r = env.get("/inspections", since=at(5).isoformat(), until=at(5).isoformat())
    assert r.status_code == 200 and r.json() == {"items": [], "total": 0}


def test_since_after_until_is_400(env):
    r = env.get("/inspections", since=at(6).isoformat(), until=at(5).isoformat())
    assert r.status_code == 400 and r.json()["error"] == "bad_request" and "since" in r.json()["message"]


@pytest.mark.parametrize("value", ["2026-10-18T09:05:00", "2026-10-18", "0", "1760745900", "yesterday", "", "2026-13-40T00:00:00Z"])
@pytest.mark.parametrize("name", ["since", "until"])
def test_bad_or_naive_times_are_400(env, name, value):
    r = env.get("/inspections", **{name: value})
    assert r.status_code == 400 and r.json()["error"] == "bad_request" and name in r.json()["message"]


def test_paging(env):
    first = env.get("/inspections", limit=5)
    assert ids(first) == [f"INS-{n:06d}" for n in range(12, 7, -1)] and first.json()["total"] == 12  # total 은 limit 과 무관
    assert ids(env.get("/inspections", limit=5, offset=10)) == ["INS-000002", "INS-000001"]
    assert env.get("/inspections", limit=5, offset=10).json()["total"] == 12
    beyond = env.get("/inspections", offset=12)
    assert beyond.status_code == 200 and beyond.json() == {"items": [], "total": 12}
    assert len(env.get("/inspections", limit=1).json()["items"]) == 1
    assert env.get("/inspections", limit=1000).status_code == 200


def test_paging_with_filter_counts_only_the_filtered_rows(env):
    r = env.get("/inspections", scenario="normal", limit=2, offset=1)
    assert r.json()["total"] == 6 and ids(r) == ["INS-000009", "INS-000007"]


@pytest.mark.parametrize("params", [
    {"limit": 0}, {"limit": 1001}, {"limit": -1}, {"limit": "abc"}, {"limit": "1.5"}, {"limit": ""},
    {"offset": -1}, {"offset": "abc"}, {"offset": "2.5"},
    {"scenario": "bogus"}, {"scenario": "NORMAL"}, {"data_split": "bogus"}, {"data_split": "dataset"},
])
def test_out_of_range_input_is_400_not_422(env, params):
    r = env.get("/inspections", **params)
    assert r.status_code == 400 and set(r.json()) == {"error", "message"} and r.json()["error"] == "bad_request"
    assert next(iter(params)) in r.json()["message"]


# ---------- 상세 ----------

def test_detail_format(env):
    r = env.get("/inspections/INS-000002")
    body = r.json()
    assert r.status_code == 200 and set(body) == ITEM_KEYS | {"predictions"}
    assert body["inspection_id"] == "INS-000002" and body["true_label"] == "NG" and body["drift_type"] == "dark"


def test_detail_predictions_are_newest_first_with_contract_fields(env):
    predictions = env.get("/inspections/INS-000002").json()["predictions"]
    assert [p["prediction_id"] for p in predictions] == ["PRD-000003", "PRD-000002", "PRD-000001"]
    assert all(set(p) == PREDICTION_KEYS and p["inspection_id"] == "INS-000002" and p["model_version"] == "v1" for p in predictions)
    assert predictions[0]["predicted_label"] == "NG" and predictions[1]["predicted_label"] == "OK"
    assert predictions[0]["score"] == 0.9 and predictions[0]["threshold"] == 0.5 and predictions[0]["latency_ms"] == 3
    assert datetime.fromisoformat(predictions[0]["predicted_at"]) == at(102)


def test_detail_without_predictions_has_an_empty_list(env):
    assert env.get("/inspections/INS-000001").json()["predictions"] == []


@pytest.mark.parametrize("missing", ["INS-999999", "nope", "ins-000001", "INS-000001%20"])
def test_unknown_inspection_is_404(env, missing):
    for suffix in ("", "/image"):
        r = env.client.get(f"/inspections/{missing}{suffix}", headers=env.headers)
        assert r.status_code == 404 and r.json()["error"] == "not_found"


# ---------- 이미지 ----------

def test_image_is_the_file_itself(env):
    r = env.get("/inspections/INS-000007/image")  # train 폴더
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.content == (env.root / "data" / "train" / "INS-000007.png").read_bytes() == png_bytes(70)
    assert r.content.startswith(b"\x89PNG")


def test_image_whose_file_is_gone_is_404(env):
    with env.scope() as s:
        InspectionRepository(s).add(make_inspection(50, image_path="data/pool/gone.png"))
    r = env.get("/inspections/INS-000050/image")
    assert r.status_code == 404 and r.json() == NOT_FOUND_IMAGE
    assert env.get("/inspections/INS-000050").status_code == 200  # 메타데이터는 그대로 조회된다


# ---------- 이미지 경로 차단 (DB 의 image_path 가 위험한 값일 때) ----------

@pytest.fixture(scope="module")
def evil(tmp_path_factory):
    e = Env(tmp_path_factory.mktemp("evil"))
    outside = tmp_path_factory.mktemp("outside_target")
    write_file(e.root, "outside.png", png_bytes())
    write_file(e.root, "parts/collector/collector.toml", b"secret")
    write_file(e.root, "data/mlops.db", b"database")
    write_file(e.root, "data/secret.key", b"k" * 64)
    write_file(e.root, "data/pool/readme.txt", b"text")
    write_file(e.root, "data/pool/INS-000001.png", png_bytes())
    write_file(outside, "secret.png", png_bytes())
    make_dir_link(e.root / "data" / "link", outside)
    make_dir_link(e.root / "data" / "alias", e.root / "data" / "pool")
    paths = {
        "INS-000100": "../outside.png", "INS-000101": "data/../outside.png", "INS-000102": "data/../parts/collector/collector.toml",
        "INS-000103": "parts/collector/collector.toml", "INS-000104": "data/mlops.db", "INS-000105": "data/secret.key",
        "INS-000106": "data/pool/readme.txt", "INS-000107": "outside.png", "INS-000108": "data/link/secret.png",
        "INS-000109": "data/alias/INS-000001.png", "INS-000110": "data\\..\\outside.png",
    }
    with e.scope() as s:
        for ins_id, path in paths.items():
            raw_inspection(s.connection(), id=ins_id, path=path)
    e.paths = paths
    yield e
    e.app.state.engine.dispose()


def test_the_normal_file_in_the_same_environment_is_served(evil):
    with evil.scope() as s:
        raw_inspection(s.connection(), id="INS-000001", path="data/pool/INS-000001.png")
    assert evil.get("/inspections/INS-000001/image").status_code == 200


@pytest.mark.parametrize("ins_id", [f"INS-{n:06d}" for n in range(100, 111)])
def test_dangerous_image_paths_are_all_404_with_the_same_message(evil, ins_id):
    r = evil.get(f"/inspections/{ins_id}/image")
    assert r.status_code == 404 and r.json() == NOT_FOUND_IMAGE  # 이유(링크, 폴더 밖 등)는 응답에 없다
    assert evil.paths[ins_id].split("/")[-1] not in r.text
    assert evil.get(f"/inspections/{ins_id}").status_code == 200  # 위험한 경로여도 상세 JSON 은 그대로
