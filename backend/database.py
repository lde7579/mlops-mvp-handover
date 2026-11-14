"""DB 연결: SQLite 엔진, 연결 설정(PRAGMA), 시각 저장 방식(UTC), 세션 묶음(session_scope).

- DB 파일은 data/mlops.db (data/ 는 .gitignore 에 있어 git 과 조원 전달 압축에서 제외된다).
- 테스트는 ":memory:" 를 쓴다.
- 시각은 UTC 고정 폭 문자열(2026-10-18T04:12:30.123456+00:00)로 저장한다. 시간대가 섞인 문자열은 정렬·기간 필터가
  틀리지만, UTC 고정 폭이면 문자열 순서가 곧 시간 순서다. 시간대 없는 시각은 거부한다.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import String, create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.types import TypeDecorator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "mlops.db"
MEMORY = ":memory:"
BUSY_TIMEOUT_MS = 5000


class Base(DeclarativeBase):
    pass


def ensure_aware(name: str, value) -> datetime:
    """시간대가 있는 datetime 만 허용한다."""
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name}: 시간대가 있는 시각이어야 합니다 (받은 값: {value!r})")
    return value


def to_utc_text(value: datetime) -> str:
    return ensure_aware("시각", value).astimezone(timezone.utc).isoformat(timespec="microseconds")


class UtcDateTime(TypeDecorator):
    """시간대 있는 datetime <-> UTC 고정 폭 문자열."""

    impl = String(32)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return None if value is None else to_utc_text(value)

    def process_result_value(self, value, dialect):
        return None if value is None else datetime.fromisoformat(value)


def create_db_engine(db_path=DEFAULT_DB_PATH):
    """SQLite 엔진. 연결할 때마다 외래키 검사를 켠다(SQLite 는 기본이 꺼져 있다)."""
    memory = str(db_path) == MEMORY
    if memory:
        # 메모리 DB 는 연결마다 따로 생기므로 연결 하나를 모든 스레드가 공유한다.
        engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    else:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{path.as_posix()}", connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        if not memory:
            cursor.execute("PRAGMA journal_mode=WAL")  # 시뮬레이터 생성과 API 조회가 동시에 접근해도 읽기가 막히지 않는다
            cursor.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")  # 다른 쓰기가 끝나길 기다리는 시간
        cursor.close()

    return engine


def init_db(engine) -> None:
    """테이블을 만든다. 이미 있는 테이블은 그대로 두고 없는 테이블만 추가한다 (마이그레이션 도구는 쓰지 않는다)."""
    from backend import models  # noqa: F401  (테이블 정의를 Base 에 등록)

    Base.metadata.create_all(engine)


def make_session_factory(engine):
    return sessionmaker(engine, expire_on_commit=False)


@contextmanager
def session_scope(factory):
    """성공하면 커밋, 오류가 나면 롤백한다. 저장소는 커밋하지 않고 이 묶음이 커밋한다."""
    session = factory()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()
