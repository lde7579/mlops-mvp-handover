"""POST /simulator/generate: 수집기로 이미지를 만들고 DB 에 등록한다.

모든 테스트는 임시 폴더를 프로젝트 루트로 쓴다 (실제 data/pool 에는 파일을 만들지 않는다 -> 맨 아래 가드 fixture 가 확인).
수집기는 이미지를 실제로 만들므로 느리다. 시험마다 만드는 이미지를 최소로 하고(전체 약 40장), 읽기만 하는 확인은
모듈 단위 fixture 가 만든 결과를 함께 쓴다.
"""
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

import backend.generation as generation_module
from backend.database import create_db_engine, session_scope
from backend.main import create_app
from backend.repository import DuplicateError, InspectionRepository
from backend.settings import DEFAULT_COLLECTOR_CONFIG, PROJECT_ROOT
from parts.collector.collector import Collector
from tests.backend.helpers import asgi_call, bearer, make_inspection, make_settings, write_file

ITEM_KEYS = {"inspection_id", "captured_at", "image_path", "true_label", "missing_parts", "scenario", "data_split", "drift_type"}
REAL_POOL = PROJECT_ROOT / "data" / "pool"


class Env:
    def __init__(self, root, db_path=":memory:"):
        self.root = Path(root)
        self.settings = make_settings(root=str(root), db_path=str(db_path))
        self.app = create_app(self.settings)
        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.headers = bearer(self.settings, "operator")

    def generate(self, json=None, headers=None, **kw):
        return self.client.post("/simulator/generate", json=json, headers=self.headers if headers is None else headers, **kw)

    @property
    def pool(self) -> Path:
        return self.root / "data" / "pool"

    def pngs(self):
        return sorted(p.name for p in self.pool.glob("*.png")) if self.pool.exists() else []

    def count(self) -> int:
        with session_scope(self.app.state.session_factory) as s:
            return InspectionRepository(s).count()

    def close(self):
        self.app.state.engine.dispose()


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.close()


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    """모듈 안에서 함께 쓰는 환경 + 먼저 만들어 둔 normal 1건, drift 1건 (이미지 2장)."""
    e = Env(tmp_path_factory.mktemp("generate_shared"))
    e.normal = e.generate()
    e.drift = e.generate({"scenario": "drift"})
    yield e
    e.close()


@pytest.fixture(autouse=True, scope="module")
def real_data_pool_is_untouched():
    """이 파일의 어떤 테스트도 실제 프로젝트의 data/pool 을 만들거나 바꾸지 않는다."""
    before = sorted(p.name for p in REAL_POOL.iterdir()) if REAL_POOL.exists() else None
    yield
    after = sorted(p.name for p in REAL_POOL.iterdir()) if REAL_POOL.exists() else None
    assert after == before, "테스트가 실제 data/pool 을 바꿨습니다"


# ---------- 인증·권한 ----------

def test_requires_login(shared):
    for headers in ({}, {"Authorization": "Bearer garbage"}):
        r = shared.generate(headers=headers)
        assert r.status_code == 401 and r.json()["error"] == "unauthorized" and r.headers["www-authenticate"] == "Bearer"


def test_401_comes_before_body_validation(shared):
    r = shared.generate({"scenario": "bogus", "x": 1}, headers={})
    assert r.status_code == 401


@pytest.mark.parametrize("user", ["operator", "admin"])
def test_operator_and_admin_can_generate(shared, user):
    assert shared.generate(headers=bearer(shared.settings, user)).status_code == 201


# ---------- 응답 형식 ----------

def test_created_response_has_the_8_contract_fields(shared):
    r = shared.normal
    body = r.json()
    assert r.status_code == 201 and set(body) == ITEM_KEYS
    assert body["data_split"] == "pool" and body["scenario"] == "normal" and body["drift_type"] is None
    assert body["true_label"] in ("OK", "NG")
    assert (body["true_label"] == "OK") == (body["missing_parts"] == [])
    assert all(1 <= p <= 6 for p in body["missing_parts"])
    assert body["image_path"] == f"data/pool/{body['inspection_id']}.png"
    assert datetime.fromisoformat(body["captured_at"]).utcoffset().total_seconds() == 0 and body["captured_at"].endswith("Z")  # UTC


def test_drift_scenario_has_a_drift_type(shared):
    body = shared.drift.json()
    assert shared.drift.status_code == 201 and body["scenario"] == "drift" and body["drift_type"] in ("dark", "noise", "shift")
    assert body["data_split"] == "pool"


@pytest.mark.parametrize("kwargs", [{}, {"json": {}}, {"json": {"scenario": "normal"}}])
def test_scenario_defaults_to_normal(env, kwargs):
    r = env.client.post("/simulator/generate", headers=env.headers, **kwargs)
    assert r.status_code == 201 and r.json()["scenario"] == "normal"


@pytest.mark.parametrize("payload", [
    {"scenario": "bogus"}, {"scenario": "NORMAL"}, {"scenario": ""}, {"scenario": None}, {"scenario": 1},
    {"scenario": "normal", "data_split": "train"}, {"drift_type": "dark"}, {"unknown": 1},
])
def test_bad_body_is_400_and_creates_nothing(env, payload):
    r = env.generate(payload)
    assert r.status_code == 400 and r.json()["error"] == "bad_request" and set(r.json()) == {"error", "message"}
    assert env.pngs() == [] and env.count() == 0


@pytest.mark.parametrize("content", [b"not json", b"[1]", b'"normal"'])
def test_non_object_body_is_400(env, content):
    r = env.client.post("/simulator/generate", content=content, headers={**env.headers, "Content-Type": "application/json"})
    assert r.status_code == 400 and r.json()["error"] == "bad_request"
    assert env.pngs() == [] and env.count() == 0


def test_extra_field_message_names_the_field(env):
    assert "unknown" in env.generate({"unknown": 1}).json()["message"]


def test_other_methods_are_405(shared):
    r = shared.client.get("/simulator/generate", headers=shared.headers)
    assert r.status_code == 405 and "POST" in r.headers["allow"]


# ---------- 파일과 DB 가 맞는다 ----------

def test_file_and_db_row_match_the_response(shared):
    body = shared.normal.json()
    path = shared.root / body["image_path"]
    with Image.open(path) as img:
        assert img.format == "PNG" and img.size == (128, 128) and img.mode == "L"
    detail = shared.client.get(f"/inspections/{body['inspection_id']}", headers=shared.headers).json()
    assert {k: detail[k] for k in ITEM_KEYS} == body and detail["predictions"] == []
    image = shared.client.get(f"/inspections/{body['inspection_id']}/image", headers=shared.headers)
    assert image.status_code == 200 and image.content == path.read_bytes()


def test_new_inspection_appears_first_in_the_list(shared):
    items = shared.client.get("/inspections", params={"data_split": "pool"}, headers=shared.headers).json()["items"]
    assert items[0]["inspection_id"] > items[-1]["inspection_id"]
    assert {shared.normal.json()["inspection_id"], shared.drift.json()["inspection_id"]} <= {i["inspection_id"] for i in items}


def test_ids_are_consecutive_from_1_and_files_match_rows(env):
    assert [env.generate().json()["inspection_id"] for _ in range(3)] == ["INS-000001", "INS-000002", "INS-000003"]
    assert env.pngs() == ["INS-000001.png", "INS-000002.png", "INS-000003.png"] and env.count() == 3
    assert [p.name for p in (env.root / "data").iterdir()] == ["pool"]  # 다른 폴더에는 쓰지 않는다


def test_same_start_gives_the_same_image(tmp_path):
    images = []
    for name in ("a", "b"):
        e = Env(tmp_path / name)
        e.generate({"scenario": "drift"})
        images.append((e.pool / "INS-000001.png").read_bytes())
        e.close()
    assert images[0] == images[1]  # 시드 = 설정 시드 + 시작 번호 -> 재현된다


# ---------- 번호: DB 최대 번호와 폴더 안 파일을 모두 고려 ----------

def test_next_number_follows_the_database_maximum(env):
    with session_scope(env.app.state.session_factory) as s:
        InspectionRepository(s).add(make_inspection(100, image_path="data/dataset/train/INS-000100.png", data_split="train"))
    assert env.generate().json()["inspection_id"] == "INS-000101"
    assert env.generate().json()["inspection_id"] == "INS-000102"


def test_number_in_the_database_is_noticed_even_after_the_collector_started(env):
    assert env.generate().json()["inspection_id"] == "INS-000001"
    with session_scope(env.app.state.session_factory) as s:  # 서버가 도는 중에 다른 경로로 큰 번호가 들어온 경우
        InspectionRepository(s).add(make_inspection(500, image_path="data/dataset/train/INS-000500.png", data_split="train"))
    assert env.generate().json()["inspection_id"] == "INS-000501"


def test_next_number_follows_existing_files_when_larger(env):
    write_file(env.root, "data/pool/INS-000200.png")
    assert env.generate().json()["inspection_id"] == "INS-000201"


def test_database_number_wins_over_smaller_file_number(env):
    write_file(env.root, "data/pool/INS-000200.png")
    with session_scope(env.app.state.session_factory) as s:
        InspectionRepository(s).add(make_inspection(300, image_path="data/dataset/train/INS-000300.png", data_split="train"))
    assert env.generate().json()["inspection_id"] == "INS-000301"


# ---------- 앱을 만들 때는 아무것도 만들지 않는다 ----------

def test_creating_the_app_and_other_requests_create_no_files(tmp_path):
    e = Env(tmp_path)
    try:
        assert list(tmp_path.iterdir()) == []
        e.client.get("/health")
        e.client.get("/inspections", headers=e.headers)
        e.generate({"scenario": "bogus"})  # 실패한 요청도 수집기를 만들지 않는다
        assert list(tmp_path.iterdir()) == []
        assert e.generate().status_code == 201 and (tmp_path / "data" / "pool").is_dir()
    finally:
        e.close()


# ---------- 실패하면 PNG 를 남기지 않는다 ----------

def test_db_failure_removes_the_png(env, monkeypatch):
    def broken_add(self, inspection):
        raise RuntimeError("DB 오류")

    monkeypatch.setattr(InspectionRepository, "add", broken_add)
    r = env.generate()
    assert r.status_code == 500 and r.json()["error"] == "server_error" and "DB" not in r.json()["message"]
    assert env.pngs() == [] and env.count() == 0


def test_commit_failure_removes_the_png_and_the_next_request_works(env, monkeypatch):
    real_commit, calls = Session.commit, []

    def failing_commit(self):
        calls.append(1)
        if len(calls) == 1:
            raise OperationalError("COMMIT", {}, Exception("disk I/O error"))
        return real_commit(self)

    monkeypatch.setattr(Session, "commit", failing_commit)
    assert env.generate().status_code == 500
    assert env.pngs() == [] and env.count() == 0
    r = env.generate()
    assert r.status_code == 201 and env.pngs() == [r.json()["inspection_id"] + ".png"] and env.count() == 1


def test_duplicate_id_is_retried_and_the_failed_png_is_removed(env, monkeypatch):
    real_add, calls = InspectionRepository.add, []

    def add_with_two_duplicates(self, inspection):
        calls.append(inspection.inspection_id)
        if len(calls) <= 2:
            raise DuplicateError("겹침")
        return real_add(self, inspection)

    monkeypatch.setattr(InspectionRepository, "add", add_with_two_duplicates)
    r = env.generate()
    assert r.status_code == 201 and len(calls) == 3
    assert env.pngs() == [r.json()["inspection_id"] + ".png"] and env.count() == 1  # 겹친 시도의 파일은 없다


def test_giving_up_after_3_duplicates_leaves_nothing(env, monkeypatch):
    calls = []

    def always_duplicate(self, inspection):
        calls.append(1)
        raise DuplicateError("겹침")

    monkeypatch.setattr(InspectionRepository, "add", always_duplicate)
    r = env.generate()
    assert r.status_code == 500 and len(calls) == generation_module.ID_ATTEMPTS == 3
    assert env.pngs() == [] and env.count() == 0


def test_png_removal_failure_is_logged_and_does_not_hide_the_original_error(env, monkeypatch, caplog):
    monkeypatch.setattr(InspectionRepository, "add", lambda self, inspection: (_ for _ in ()).throw(RuntimeError("DB 오류")))
    real_unlink = Path.unlink

    def failing_unlink(self, *args, **kwargs):
        if self.suffix == ".png":
            raise PermissionError("잠겨 있음")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    with caplog.at_level("WARNING", logger="backend"):
        r = env.generate()
    assert r.status_code == 500  # 원래 오류(500)가 그대로 나간다
    assert "PNG 를 지우지 못했습니다" in caplog.text


def test_collector_failure_is_500_and_the_lock_is_released(env, monkeypatch):
    real = Collector.next_sample
    monkeypatch.setattr(Collector, "next_sample", lambda self, **kw: (_ for _ in ()).throw(RuntimeError("수집 실패")))
    assert env.generate().status_code == 500 and env.pngs() == [] and env.count() == 0
    monkeypatch.setattr(Collector, "next_sample", real)
    done = threading.Event()
    threading.Thread(target=lambda: (env.generate(), done.set()), daemon=True).start()
    assert done.wait(20), "락이 풀리지 않아 다음 요청이 멈췄습니다"
    assert env.count() == 1


# ---------- 동시 요청 ----------

def test_concurrent_requests_get_unique_ids_and_leave_no_orphans(tmp_path):
    e = Env(tmp_path, db_path=tmp_path / "test.db")  # 여러 스레드가 쓰므로 파일 DB
    try:
        with ThreadPoolExecutor(8) as pool:
            batches = list(pool.map(lambda _: [e.generate() for _ in range(3)], range(8)))
        responses = [r for batch in batches for r in batch]
        assert [r.status_code for r in responses] == [201] * 24
        got = sorted(r.json()["inspection_id"] for r in responses)
        assert got == [f"INS-{n:06d}" for n in range(1, 25)]  # 겹침도 빠짐도 없다
        assert e.pngs() == [f"{i}.png" for i in got] and e.count() == 24
    finally:
        e.close()


# ---------- 응답이 나가기 전에 이미 커밋되어 있다 ----------

def test_row_is_committed_before_the_response_starts(tmp_path):
    e = Env(tmp_path, db_path=tmp_path / "test.db")
    other = create_db_engine(tmp_path / "test.db")  # 다른 연결 (커밋되지 않은 데이터는 보이지 않는다)
    seen = []
    try:
        def on_start(status):
            with other.connect() as conn:
                seen.append((status, conn.exec_driver_sql("SELECT COUNT(*) FROM inspection").scalar()))

        asgi_call(e.app, "POST", "/simulator/generate", on_start, headers=e.headers)
        assert seen == [(201, 1)]
    finally:
        other.dispose()
        e.close()


# ---------- 수집기 내부 속성 _next_id 에 의존하는 부분을 지킨다 ----------
# backend/generation.py 는 수집기의 _next_id 를 직접 읽고 올린다 (parts/ 를 고치지 않기 위한 임시 방법).
# 수집기가 바뀌어 이 속성의 이름이나 의미가 달라지면 아래 시험이 바로 깨져서 알려 준다.

def test_collector_still_has_the_next_id_attribute_we_depend_on(tmp_path):
    c = Collector(DEFAULT_COLLECTOR_CONFIG, root=tmp_path)
    assert isinstance(getattr(c, "_next_id", None), int) and not isinstance(c._next_id, bool), \
        "수집기에 _next_id 가 없어졌거나 정수가 아닙니다 -> backend/generation.py 의 _ready() 를 고쳐야 합니다"
    assert c._next_id == 1  # 빈 폴더에서는 1 부터
    assert callable(getattr(c, "_last_saved_id", None))


def test_setting_next_id_decides_the_next_inspection_number(tmp_path):
    c = Collector(DEFAULT_COLLECTOR_CONFIG, root=tmp_path)
    c._next_id = 100
    sample = c.next_sample(scenario="normal", data_split="pool", label="OK")
    assert sample.inspection.inspection_id == "INS-000100" and (tmp_path / "data" / "pool" / "INS-000100.png").is_file(), \
        "_next_id 를 바꿔도 번호가 바뀌지 않습니다 -> backend/generation.py 의 방식이 더는 맞지 않습니다"
    assert c._next_id == 101  # 저장에 성공하면 하나 올라간다


def test_service_actually_goes_through_next_id(env, monkeypatch):
    seen = []
    real = Collector.next_sample

    def spy(self, **kw):
        seen.append(self._next_id)
        return real(self, **kw)

    monkeypatch.setattr(Collector, "next_sample", spy)
    with session_scope(env.app.state.session_factory) as s:
        InspectionRepository(s).add(make_inspection(41, image_path="data/dataset/train/INS-000041.png", data_split="train"))
    env.generate()
    assert seen == [42]
