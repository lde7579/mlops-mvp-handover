"""InspectionRepository.max_id(): 새 검사 번호를 정할 때 쓰는 'DB 에서 가장 큰 INS 번호'."""
import pytest

from backend.repository import InspectionRepository
from tests.backend.helpers import make_inspection, raw_inspection


def test_empty_table_gives_zero(session):
    assert InspectionRepository(session).max_id() == 0


def test_gives_the_largest_number_not_the_latest_insert(session):
    repo = InspectionRepository(session)
    for n in (5, 100, 7):
        repo.add(make_inspection(n))
    assert repo.max_id() == 100
    assert isinstance(repo.max_id(), int)


def test_sees_rows_added_in_the_same_uncommitted_session(session):
    repo = InspectionRepository(session)
    repo.add(make_inspection(3))
    assert repo.max_id() == 3
    repo.add(make_inspection(4))
    assert repo.max_id() == 4


def test_compares_numbers_not_strings_beyond_six_digits(session):
    conn = session.connection()
    raw_inspection(conn, id="INS-999999", path="data/pool/INS-999999.png")
    raw_inspection(conn, id="INS-1000000", path="data/pool/INS-1000000.png")
    assert InspectionRepository(session).max_id() == 1_000_000  # 글자 순서라면 INS-999999 가 더 크다


@pytest.mark.parametrize("odd_id", ["legacy-5999", "INS-", "INS-12x", "XINS-9999", "ins-9999", "INS--9999", "INS- 9999", "9999"])
def test_ids_in_other_formats_are_ignored(session, odd_id):
    conn = session.connection()
    raw_inspection(conn, id="INS-000002", path="data/pool/INS-000002.png")
    raw_inspection(conn, id=odd_id, path="data/pool/odd.png")
    assert InspectionRepository(session).max_id() == 2


def test_only_odd_ids_gives_zero(session):
    raw_inspection(session.connection(), id="legacy-1", path="data/pool/legacy-1.png")
    assert InspectionRepository(session).max_id() == 0
