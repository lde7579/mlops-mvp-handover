"""DB 세션 의존성(SessionDep): 요청마다 세션 하나, 커밋 시점, 롤백.

핵심: 쓰기 응답이 클라이언트에 나가는 순간, 다른 연결에서 그 데이터가 보여야 한다. TestClient 는 앱이 완전히 끝날
때까지 기다려서 이 차이를 보여 주지 못하므로, 앱을 ASGI 로 직접 호출해 "응답 시작" 순간에 다른 엔진으로 읽는다.
"""
import itertools
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Annotated

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.database import create_db_engine
from backend.deps import SessionDep, get_session
from backend.main import create_app
from tests.backend.helpers import asgi_call, make_settings

INSERT = text("INSERT INTO inspection VALUES (:i, '2026-10-18T00:00:00.000000+00:00', 'data/a.png', 'NG', '[2]', "
              "'normal', NULL, 'pool')")
REPEAT = 50


@pytest.fixture
def env(tmp_path):
    path = tmp_path / "session.db"  # 파일 DB (WAL): 다른 연결이 커밋된 것만 볼 수 있다
    app = create_app(make_settings(db_path=str(path)))
    reader = create_db_engine(path)  # 앱과 별개인 엔진 = 별개의 연결
    counter, seen = itertools.count(1), []

    def write(session):
        row_id = f"INS-{next(counter):06d}"
        session.execute(INSERT, {"i": row_id})
        return row_id

    @app.post("/_test/write_function")
    def write_function(session: SessionDep):
        seen.append(session)
        return {"id": write(session)}

    @app.post("/_test/write_request")  # scope 를 지정하지 않은 기본값
    def write_request(session: Annotated[Session, Depends(get_session)]):
        return {"id": write(session)}

    @app.post("/_test/fail_function")
    def fail_function(session: SessionDep):
        write(session)
        raise RuntimeError("일부러 실패")

    @app.get("/_test/count")
    def count(session: SessionDep):
        return {"count": session.scalar(text("SELECT count(*) FROM inspection"))}

    def rows():
        with reader.connect() as c:
            return c.exec_driver_sql("SELECT count(*) FROM inspection").scalar()

    def exists(row_id):
        with reader.connect() as c:
            return c.exec_driver_sql("SELECT count(*) FROM inspection WHERE inspection_id = ?", (row_id,)).scalar() == 1

    yield SimpleNamespace(app=app, rows=rows, exists=exists, seen=seen,
                          client=TestClient(app, raise_server_exceptions=False))
    app.state.engine.dispose()
    reader.dispose()


def test_session_dependency_runs_its_exit_code_before_the_response():
    assert SessionDep.__metadata__[0].scope == "function"


def test_each_request_gets_its_own_session(env):
    for _ in range(3):
        assert env.client.post("/_test/write_function").status_code == 200
    assert len(env.seen) == 3 and len({id(s) for s in env.seen}) == 3


def test_successful_request_is_committed(env):
    row_id = env.client.post("/_test/write_function").json()["id"]
    assert env.exists(row_id) and env.rows() == 1


def test_exception_rolls_back_and_returns_500(env):
    r = env.client.post("/_test/fail_function")
    assert r.status_code == 500 and r.json()["error"] == "server_error"
    assert env.rows() == 0
    assert env.client.post("/_test/write_function").status_code == 200  # 실패 뒤에도 정상 동작
    assert env.rows() == 1


def test_session_is_closed_after_the_request(env):
    env.client.post("/_test/write_function")
    assert not env.seen[0].in_transaction()


def test_write_is_visible_to_another_connection_when_the_response_starts(env):
    """쓰기 응답 직후 다른 요청(연결)이 읽으면 보인다 (50회 반복)."""
    outcomes = []
    for _ in range(REPEAT):
        before = env.rows()
        asgi_call(env.app, "POST", "/_test/write_function",
                  lambda status, b=before: outcomes.append((status, env.rows() - b)))
    assert outcomes == [(200, 1)] * REPEAT  # 응답이 나가는 순간 이미 커밋되어 새 행 1개가 보인다
    assert env.rows() == REPEAT


def test_contrast_default_scope_commits_only_after_the_response_starts(env):
    """대조: scope 를 지정하지 않은 기본값은 응답을 보낸 뒤에 커밋한다. 그래서 SessionDep 는 scope='function' 을 쓴다."""
    outcomes = []
    for _ in range(REPEAT):
        before = env.rows()
        asgi_call(env.app, "POST", "/_test/write_request",
                  lambda status, b=before: outcomes.append((status, env.rows() - b)))
        assert env.rows() == before + 1  # 앱 호출이 끝나면 커밋되어 있다
    assert outcomes == [(200, 0)] * REPEAT  # 응답이 나가는 순간에는 아직 안 보인다


def test_writes_and_reads_stay_consistent_under_concurrent_requests(env):
    threads, per_thread = 8, 10

    def work():
        client = TestClient(env.app, raise_server_exceptions=False)
        missing = []
        for _ in range(per_thread):
            row_id = client.post("/_test/write_function").json()["id"]
            if not env.exists(row_id):  # 응답을 받은 직후 다른 연결에서 읽는다
                missing.append(row_id)
        return missing

    with ThreadPoolExecutor(max_workers=threads) as pool:
        results = [f.result() for f in [pool.submit(work) for _ in range(threads)]]
    assert results == [[]] * threads
    assert env.rows() == threads * per_thread
    assert env.client.get("/_test/count").json() == {"count": threads * per_thread}
