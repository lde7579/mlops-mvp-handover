"""PredictionRepository: ID 발급, 검증, 필터·정렬·페이징, 읽기 검증, 동시 추가."""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest

from backend.database import create_db_engine, init_db, make_session_factory, session_scope
from backend.repository import (
    InspectionRepository,
    ModelVersionRepository,
    NotFoundError,
    PredictionRepository,
    Stage,
)
from common.schemas import Label, SchemaError
from tests.backend.helpers import (
    at,
    make_inspection,
    make_metrics,
    make_output,
    raw_inspection,
    raw_prediction,
    seed,
)


@pytest.fixture
def repo(session):
    seed(session, inspections=3, versions=("v1", "v2"))  # INS-1·2 는 NG, INS-3 은 OK / v1 production, v2 candidate
    return PredictionRepository(session)


def add(repo, n=1, label="NG", score=0.9, version="v1", when=0, threshold=0.5):
    return repo.add(make_output(label, score, version), inspection_id=f"INS-{n:06d}", threshold=threshold, predicted_at=at(when))


def ids(items):
    return [p.prediction_id for p in items]


def test_ids_are_issued_in_order_starting_at_one(repo):
    got = [add(repo, 1).prediction_id for _ in range(3)]
    assert got == ["PRD-000001", "PRD-000002", "PRD-000003"]


def test_returned_prediction_equals_what_is_read_back(repo):
    p = add(repo, 2, label="NG", score=0.8, version="v2", when=5, threshold=0.7)
    assert p == repo.get(p.prediction_id)
    assert (p.inspection_id, p.model_version, p.predicted_label, p.score, p.threshold, p.latency_ms) \
        == ("INS-000002", "v2", Label.NG, 0.8, 0.7, 3)
    assert p.predicted_at == at(5) and p.predicted_at.utcoffset() == timedelta(0)  # UTC
    assert repo.get("PRD-999999") is None


def test_threshold_from_the_caller_decides_the_label_check(repo):
    assert add(repo, 1, label="OK", score=0.7, threshold=0.8).predicted_label is Label.OK  # 0.7 < 0.8
    assert add(repo, 1, label="NG", score=0.5, threshold=0.5).predicted_label is Label.NG  # 경계값은 NG


def test_label_that_disagrees_with_score_is_rejected_and_nothing_is_written(repo):
    with pytest.raises(SchemaError):
        add(repo, 1, label="NG", score=0.3, threshold=0.5)
    with pytest.raises(SchemaError):
        add(repo, 1, label="OK", score=0.7, threshold=0.5)
    assert repo.count() == 0
    assert add(repo, 1).prediction_id == "PRD-000001"  # 실패한 시도가 번호를 쓰지 않았다


def test_unknown_inspection_or_model_version_raises_not_found_and_repo_stays_usable(repo):
    with pytest.raises(NotFoundError, match="inspection_id"):
        add(repo, 99)
    with pytest.raises(NotFoundError, match="model_version"):
        add(repo, 1, version="v9")
    assert repo.count() == 0
    assert add(repo, 1).prediction_id == "PRD-000001"


def test_time_without_timezone_is_rejected(repo):
    from datetime import datetime
    with pytest.raises(ValueError, match="시간대"):
        repo.add(make_output(), inspection_id="INS-000001", threshold=0.5, predicted_at=datetime(2026, 10, 18, 9, 0))
    assert repo.count() == 0


@pytest.fixture
def mixed(repo):
    """INS-1(NG)·INS-2(NG)·INS-3(OK) 에 대한 판정: 1 정판, 2 오판(OK), 3 오판(NG), 다시 1 오판(v2)."""
    a = add(repo, 1, "NG", 0.9, "v1", when=1)     # 정판
    b = add(repo, 2, "OK", 0.2, "v1", when=2)     # 오판 (정답 NG)
    c = add(repo, 3, "NG", 0.8, "v2", when=3)     # 오판 (정답 OK)
    d = add(repo, 1, "OK", 0.1, "v2", when=4)     # 오판 (정답 NG)
    return repo, (a, b, c, d)


def test_default_order_is_newest_first(mixed):
    repo, (a, b, c, d) = mixed
    assert ids(repo.list()) == ids([d, c, b, a])


def test_wrong_filter_uses_the_true_label(mixed):
    repo, (a, b, c, d) = mixed
    assert ids(repo.list(wrong=True)) == ids([d, c, b])
    assert ids(repo.list(wrong=False)) == ids([a])
    assert repo.count(wrong=True) == 3 and repo.count(wrong=False) == 1 and repo.count() == 4


def test_model_version_and_inspection_filters(mixed):
    repo, (a, b, c, d) = mixed
    assert ids(repo.list(model_version="v1")) == ids([b, a])
    assert ids(repo.list(model_version="v2")) == ids([d, c])
    assert ids(repo.list(inspection_id="INS-000001")) == ids([d, a])
    assert ids(repo.list(model_version="v2", wrong=True, inspection_id="INS-000001")) == ids([d])
    assert repo.list(model_version="v9") == []
    assert repo.count(model_version="v2") == 2 and repo.count(inspection_id="INS-000001") == 2


def test_same_time_orders_by_id_descending(repo):
    made = [add(repo, 1, when=0) for _ in range(3)]
    assert ids(repo.list()) == ids(reversed(made))


def test_paging(mixed):
    repo, (a, b, c, d) = mixed
    assert ids(repo.list(limit=2)) == ids([d, c])
    assert ids(repo.list(limit=2, offset=2)) == ids([b, a])
    assert ids(repo.list(limit=10, offset=3)) == ids([a])
    assert repo.list(offset=4) == []


@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"limit": 1001}, {"limit": True}, {"offset": -1}, {"wrong": "yes"}, {"wrong": 1}])
def test_bad_arguments_are_rejected(mixed, kwargs):
    repo, _ = mixed
    with pytest.raises(ValueError):
        repo.list(**kwargs)


def test_a_row_that_breaks_the_label_rule_is_caught_when_read(repo, engine, session):
    # DB 는 이 규칙을 한 번 더 두지 않는다. 대신 읽을 때 스키마 검증이 계약 위반 행을 걸러낸다.
    session.commit()
    with engine.begin() as c:
        raw_prediction(c, id="PRD-000050", ins="INS-000001", ver="v1", label="NG", score=0.1, threshold=0.5)
    with pytest.raises(SchemaError):
        repo.get("PRD-000050")
    with pytest.raises(SchemaError):
        repo.list()


def test_id_overflow_is_reported_and_rolled_back(repo, engine, session):
    session.commit()
    with engine.begin() as c:
        raw_prediction(c, id="PRD-999999")
    with pytest.raises(OverflowError):
        add(repo, 1)
    assert repo.count() == 1  # 한도를 넘은 행은 남지 않는다


# ---------- 동시에 추가해도 ID 가 겹치지 않는다 (파일 DB, WAL) ----------

THREADS = 8
PER_THREAD = 25


def prepare(path):
    engine = create_db_engine(path)
    init_db(engine)
    factory = make_session_factory(engine)
    with session_scope(factory) as s:
        s.add_all([])
        ModelVersionRepository(s).register(make_metrics("v1"), stage=Stage.PRODUCTION)
        InspectionRepository(s).add(make_inspection(1))
    return engine, factory


def add_many(factory, count):
    made = []
    for i in range(count):
        with session_scope(factory) as s:  # 요청 하나 = 트랜잭션 하나 (추가가 첫 쓰기)
            made.append(PredictionRepository(s).add(
                make_output(), inspection_id="INS-000001", threshold=0.5, predicted_at=at(i)).prediction_id)
    return made


def expected_ids(total):
    return {f"PRD-{n:06d}" for n in range(1, total + 1)}


def test_concurrent_adds_from_eight_threads_have_unique_contiguous_ids(tmp_path):
    engine, factory = prepare(tmp_path / "concurrent.db")
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futures = [pool.submit(add_many, factory, PER_THREAD) for _ in range(THREADS)]
        results = [f.result() for f in futures]  # 오류가 있으면 여기서 그대로 올라온다 (오류 0)
    flat = [i for r in results for i in r]
    total = THREADS * PER_THREAD  # 200
    assert len(flat) == total and len(set(flat)) == total  # 중복 없음
    assert set(flat) == expected_ids(total)  # PRD-000001 ~ PRD-000200 빠짐없이
    with session_scope(factory) as s:
        assert PredictionRepository(s).count() == total
    engine.dispose()


def test_concurrent_adds_through_two_separate_engines_have_unique_contiguous_ids(tmp_path):
    path = tmp_path / "two_engines.db"
    engine_a, factory_a = prepare(path)
    engine_b = create_db_engine(path)  # 같은 파일을 다른 연결 묶음(엔진)으로 연다
    factory_b = make_session_factory(engine_b)
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futures = [pool.submit(add_many, factory_a if i % 2 == 0 else factory_b, PER_THREAD) for i in range(THREADS)]
        results = [f.result() for f in futures]
    flat = [i for r in results for i in r]
    total = THREADS * PER_THREAD
    assert len(flat) == total and len(set(flat)) == total
    assert set(flat) == expected_ids(total)
    with session_scope(factory_b) as s:
        assert PredictionRepository(s).count() == total
    engine_a.dispose()
    engine_b.dispose()
