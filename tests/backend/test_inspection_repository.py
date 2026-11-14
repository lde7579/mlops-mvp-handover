"""InspectionRepository: 왕복, 중복, 시각 규칙, 필터·기간·정렬·페이징."""
from datetime import datetime, timedelta, timezone

import pytest

from backend.repository import DuplicateError, InspectionRepository
from common.schemas import SchemaError
from tests.backend.helpers import KST, at, make_inspection, raw_inspection


@pytest.fixture
def repo(session):
    return InspectionRepository(session)


def test_round_trip_ng_and_drift_samples(repo):
    ng = make_inspection(1, missing_parts=[2, 5])
    drift = make_inspection(2, true_label="OK", missing_parts=[], scenario="drift", drift_type="dark", data_split="test_drift")
    repo.add(ng)
    repo.add(drift)
    assert repo.get("INS-000001") == ng and repo.get("INS-000001").missing_parts == [2, 5]
    assert repo.get("INS-000002") == drift and repo.get("INS-000002").drift_type == "dark"
    assert repo.get("INS-000001").drift_type is None
    assert repo.get("INS-999999") is None and not repo.exists("INS-999999") and repo.exists("INS-000001")


def test_time_comes_back_as_the_same_instant_in_utc(repo):
    repo.add(make_inspection(1, captured_at=datetime(2026, 10, 18, 9, 0, 0, 250000, tzinfo=KST)))
    got = repo.get("INS-000001").captured_at
    assert got == datetime(2026, 10, 18, 0, 0, 0, 250000, tzinfo=timezone.utc)
    assert got.utcoffset() == timedelta(0)


def test_duplicate_id_is_rejected_and_first_row_is_kept(repo):
    first = make_inspection(1, missing_parts=[3])
    repo.add(first)
    with pytest.raises(DuplicateError):
        repo.add(make_inspection(1, missing_parts=[4]))
    assert repo.count() == 1 and repo.get("INS-000001") == first


def test_time_without_timezone_is_rejected_and_nothing_is_stored(repo):
    naive = make_inspection(1, captured_at=datetime(2026, 10, 18, 9, 0))  # 스키마는 허용하지만 저장소는 거부한다
    with pytest.raises(ValueError, match="시간대"):
        repo.add(naive)
    assert repo.count() == 0


def test_invalid_rows_cannot_be_created_and_bad_rows_in_db_fail_on_read(repo, engine, session):
    with pytest.raises(SchemaError):
        make_inspection(1, true_label="OK", missing_parts=[2])  # OK 인데 missing_parts 가 있음
    with pytest.raises(SchemaError):
        make_inspection(1, image_path="/abs/a.png")  # 절대경로
    session.flush()
    session.commit()
    with engine.begin() as c:  # DB 에 직접 넣은 계약 위반 행은 읽을 때 걸러진다
        raw_inspection(c, id="INS-000007", label="OK", parts="[2]")
        raw_inspection(c, id="INS-000008", path="C:\\abs\\a.png")
    for bad_id in ("INS-000007", "INS-000008"):
        with pytest.raises(SchemaError):
            repo.get(bad_id)


@pytest.fixture
def filled(repo):
    """INS-1..6: 홀수는 normal/pool, 짝수는 drift/test_drift. 시각은 at(1)..at(6)."""
    for n in range(1, 7):
        drift = n % 2 == 0
        repo.add(make_inspection(n, scenario="drift" if drift else "normal", drift_type="noise" if drift else None,
                                 data_split="test_drift" if drift else "pool"))
    return repo


def ids(items):
    return [i.inspection_id for i in items]


def test_default_order_is_newest_first(filled):
    assert ids(filled.list()) == [f"INS-{n:06d}" for n in (6, 5, 4, 3, 2, 1)]


def test_filters_by_scenario_and_data_split(filled):
    assert ids(filled.list(scenario="drift")) == ["INS-000006", "INS-000004", "INS-000002"]
    assert ids(filled.list(scenario="normal")) == ["INS-000005", "INS-000003", "INS-000001"]
    assert ids(filled.list(data_split="test_drift")) == ["INS-000006", "INS-000004", "INS-000002"]
    assert filled.list(scenario="drift", data_split="pool") == []
    assert filled.count(scenario="drift") == 3 and filled.count(data_split="pool") == 3 and filled.count() == 6


def test_period_start_is_included_and_end_is_excluded(filled):
    assert ids(filled.list(since=at(2), until=at(4))) == ["INS-000003", "INS-000002"]  # at(2) 포함, at(4) 제외
    assert ids(filled.list(since=at(5))) == ["INS-000006", "INS-000005"]
    assert ids(filled.list(until=at(2))) == ["INS-000001"]
    assert filled.count(since=at(2), until=at(4)) == 2
    # 다른 시간대로 적어도 같은 시각이면 같은 결과
    assert ids(filled.list(since=at(2).astimezone(timezone.utc), until=at(4).astimezone(timezone(timedelta(hours=-5))))) \
        == ["INS-000003", "INS-000002"]


def test_same_time_orders_by_id_descending(repo):
    for n in (1, 2, 3):
        repo.add(make_inspection(n, captured_at=at(0)))
    assert ids(repo.list()) == ["INS-000003", "INS-000002", "INS-000001"]


def test_paging(filled):
    assert ids(filled.list(limit=2)) == ["INS-000006", "INS-000005"]
    assert ids(filled.list(limit=2, offset=2)) == ["INS-000004", "INS-000003"]
    assert ids(filled.list(limit=10, offset=5)) == ["INS-000001"]
    assert filled.list(offset=6) == []


@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"limit": 1001}, {"limit": -1}, {"limit": True}, {"limit": "5"},
                                    {"offset": -1}, {"offset": 1.5}, {"offset": True}])
def test_bad_paging_is_rejected(filled, kwargs):
    with pytest.raises(ValueError):
        filled.list(**kwargs)


@pytest.mark.parametrize("kwargs", [{"scenario": "other"}, {"data_split": "validation"},
                                    {"since": datetime(2026, 10, 18)}, {"until": "2026-10-18T09:00:00+09:00"}])
def test_bad_filters_are_rejected(filled, kwargs):
    with pytest.raises(ValueError):
        filled.list(**kwargs)
    with pytest.raises(ValueError):
        filled.count(**kwargs)
