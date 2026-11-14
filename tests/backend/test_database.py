"""backend/database.py, backend/models.py: 연결 설정, 시각 저장 방식, 테이블이 계약 필드와 1:1인지."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Thread

import pytest
from sqlalchemy import inspect

from backend.database import (
    DEFAULT_DB_PATH,
    PROJECT_ROOT,
    UtcDateTime,
    create_db_engine,
    ensure_aware,
    init_db,
    session_scope,
    to_utc_text,
)
from backend.repository import InspectionRepository
from tests.backend.helpers import KST, at, make_inspection

CONTRACT_COLUMNS = {
    "inspection": {"inspection_id", "captured_at", "image_path", "true_label", "missing_parts", "scenario", "drift_type", "data_split"},
    "prediction": {"prediction_id", "inspection_id", "model_version", "predicted_label", "score", "threshold", "latency_ms", "predicted_at"},
    "model_version": {"version", "stage", "metrics", "train_data_snapshot", "created_at", "approved_by", "approved_at"},
    # 파트 3 보조 테이블 (계약 6~7장, v1.2 C7). training_run 의 step·error_message·gate_result 는 C7 로 계약에 들어왔다.
    "approval_request": {"approval_id", "model_version", "requested_by", "status", "reason", "handled_by", "handled_at", "created_at"},
    "training_run": {"run_id", "status", "trigger", "started_at", "finished_at", "result_version", "step", "error_message", "gate_result"},
    "deployment": {"deployment_id", "version", "action", "reason", "actor", "at"},
    "audit_log": {"log_id", "actor", "role", "action", "target", "at"},
}


def test_tables_match_contract_fields_one_to_one(engine):
    insp = inspect(engine)
    assert set(insp.get_table_names()) == set(CONTRACT_COLUMNS)  # 테이블은 7개뿐
    for table, expected in CONTRACT_COLUMNS.items():
        assert {c["name"] for c in insp.get_columns(table)} == expected, table


def test_indexes_exist_including_single_production_guard(engine):
    insp = inspect(engine)
    names = {i["name"]: i for t in CONTRACT_COLUMNS for i in insp.get_indexes(t)}
    for expected in ("ix_inspection_data_split", "ix_inspection_captured_at", "ix_prediction_inspection_id",
                     "ix_prediction_model_version", "ix_prediction_predicted_at"):
        assert expected in names
    assert names["uq_model_version_one_production"]["unique"]
    assert names["uq_approval_one_pending_per_version"]["unique"]


def test_foreign_keys_are_on_for_memory_and_file_databases(engine, file_engine):
    for e in (engine, file_engine):
        with e.connect() as conn:
            assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1


def test_file_database_uses_wal_and_busy_timeout_and_creates_folder(tmp_path):
    e = create_db_engine(tmp_path / "new_folder" / "a.db")
    with e.connect() as conn:
        assert conn.exec_driver_sql("PRAGMA journal_mode").scalar() == "wal"
        assert conn.exec_driver_sql("PRAGMA busy_timeout").scalar() == 5000
    assert (tmp_path / "new_folder" / "a.db").is_file()
    e.dispose()


def test_memory_database_is_shared_across_threads(engine):
    seen = []

    def read():
        with engine.connect() as conn:
            seen.append(conn.exec_driver_sql("SELECT count(*) FROM sqlite_master WHERE type = 'table'").scalar())

    t = Thread(target=read)
    t.start()
    t.join()
    assert seen == [7]  # 다른 스레드에서도 같은 DB(테이블 7개)가 보인다


def test_default_path_is_under_gitignored_data_folder_and_tests_do_not_use_it(engine, file_engine):
    assert DEFAULT_DB_PATH == PROJECT_ROOT / "data" / "mlops.db"
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8-sig").splitlines()
    assert "data/" in gitignore
    assert str(engine.url) == "sqlite://"
    assert Path(file_engine.url.database).resolve() != DEFAULT_DB_PATH.resolve()


def test_init_db_twice_is_harmless(engine):
    init_db(engine)
    assert set(inspect(engine).get_table_names()) == set(CONTRACT_COLUMNS)


# ---------- 시각: UTC 고정 폭 문자열 ----------

def test_time_is_stored_as_utc_fixed_width_text():
    text = to_utc_text(datetime(2026, 10, 18, 9, 0, 0, 123, tzinfo=KST))
    assert text == "2026-10-18T00:00:00.000123+00:00"
    assert len(text) == 32 and len(to_utc_text(at(0))) == 32


def test_time_round_trip_gives_same_instant_in_utc():
    original = datetime(2026, 10, 18, 9, 30, 15, 500000, tzinfo=KST)
    stored = UtcDateTime().process_bind_param(original, None)
    back = UtcDateTime().process_result_value(stored, None)
    assert back == original and back.utcoffset() == timedelta(0)
    assert UtcDateTime().process_bind_param(None, None) is None and UtcDateTime().process_result_value(None, None) is None


@pytest.mark.parametrize("value", [datetime(2026, 10, 18, 9, 0), "2026-10-18T09:00:00+09:00", None, 5])
def test_time_without_timezone_or_wrong_type_is_rejected(value):
    with pytest.raises(ValueError):
        ensure_aware("시각", value)
    with pytest.raises(ValueError):
        to_utc_text(value)


def test_mixed_timezones_sort_by_real_time(session):
    # 문자열 그대로 비교하면 09:00+09:00 이 01:00+00:00 보다 뒤지만, 실제로는 01:00Z 가 더 늦다
    early = make_inspection(1, captured_at=datetime(2026, 10, 18, 9, 0, tzinfo=KST))      # 00:00Z
    late = make_inspection(2, captured_at=datetime(2026, 10, 18, 1, 0, tzinfo=timezone.utc))  # 01:00Z
    repo = InspectionRepository(session)
    repo.add(early)
    repo.add(late)
    assert [i.inspection_id for i in repo.list()] == ["INS-000002", "INS-000001"]  # 최신(01:00Z) 먼저


def test_session_scope_commits_on_success_and_rolls_back_on_error(factory):
    with session_scope(factory) as s:
        InspectionRepository(s).add(make_inspection(1))
    with pytest.raises(RuntimeError):
        with session_scope(factory) as s:
            InspectionRepository(s).add(make_inspection(2))
            raise RuntimeError("중간에 실패")
    with session_scope(factory) as s:
        repo = InspectionRepository(s)
        assert repo.exists("INS-000001") and not repo.exists("INS-000002")
