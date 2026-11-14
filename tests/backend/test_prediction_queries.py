"""PredictionRepository.list_with_truth() 와 stats(): 정답을 함께 읽는 조회와 혼동행렬 집계.

시드는 tests/backend/helpers.py 의 PREDICTION_SEED (손으로 계산한 표가 주석에 있다). 읽기만 하므로 모듈 안에서 DB 하나를 함께 쓴다.
"""
import pytest

from backend.database import create_db_engine, init_db, make_session_factory, session_scope
from backend.repository import PredictionRepository, PredictionStats
from tests.backend.helpers import seed_read_only_data

ALL_IDS_NEWEST_FIRST = [f"PRD-{n:06d}" for n in range(16, 0, -1)]


@pytest.fixture(scope="module")
def seeded_factory():
    engine = create_db_engine(":memory:")
    init_db(engine)
    f = make_session_factory(engine)
    with session_scope(f) as s:
        seed_read_only_data(s)
    yield f
    engine.dispose()


@pytest.fixture
def repo(seeded_factory):
    with session_scope(seeded_factory) as s:
        yield PredictionRepository(s)


def ids(items):
    return [i.prediction.prediction_id for i in items]


# ---------- list_with_truth ----------

def test_newest_first_and_ties_by_id_descending(repo):
    assert ids(repo.list_with_truth()) == ALL_IDS_NEWEST_FIRST  # PRD-15·16 은 같은 시각 -> ID 큰 것이 먼저


def test_truth_and_correct_are_computed_from_the_inspection(repo):
    by_id = {i.prediction.prediction_id: i for i in repo.list_with_truth()}
    expected = {  # id: (정답, 판정, 정오)
        "PRD-000001": ("NG", "NG", True), "PRD-000003": ("NG", "OK", False), "PRD-000005": ("OK", "NG", False),
        "PRD-000008": ("NG", "OK", False), "PRD-000012": ("OK", "NG", False), "PRD-000014": ("OK", "OK", True),
        "PRD-000015": ("NG", "NG", True), "PRD-000016": ("OK", "NG", False),
    }
    for pid, (truth, predicted, correct) in expected.items():
        item = by_id[pid]
        assert (item.true_label.value, item.prediction.predicted_label.value, item.correct) == (truth, predicted, correct), pid


@pytest.mark.parametrize("filters", [
    {}, {"model_version": "v1"}, {"model_version": "v2"}, {"inspection_id": "INS-000007"}, {"inspection_id": "INS-000011"},
    {"wrong": True}, {"wrong": False}, {"model_version": "v1", "wrong": True}, {"inspection_id": "INS-000007", "wrong": True},
    {"model_version": "nope"},
])
def test_same_rows_and_order_as_list_and_count(repo, filters):
    with_truth = repo.list_with_truth(**filters)
    assert [i.prediction for i in with_truth] == repo.list(**filters)
    assert len(with_truth) == repo.count(**filters)


@pytest.mark.parametrize("wrong, expected", [
    (True, ["PRD-000016", "PRD-000012", "PRD-000008", "PRD-000005", "PRD-000003"]),  # FP, FP, FN, FP, FN
    (False, [f"PRD-{n:06d}" for n in (15, 14, 13, 11, 10, 9, 7, 6, 4, 2, 1)]),
])
def test_wrong_filter(repo, wrong, expected):
    got = repo.list_with_truth(wrong=wrong)
    assert ids(got) == expected and all(i.correct is (not wrong) for i in got)


def test_inspection_with_several_predictions(repo):
    assert ids(repo.list_with_truth(inspection_id="INS-000011")) == ["PRD-000013", "PRD-000011"]
    assert ids(repo.list_with_truth(inspection_id="INS-000007")) == ["PRD-000016", "PRD-000014", "PRD-000004"]


def test_paging(repo):
    assert ids(repo.list_with_truth(limit=3, offset=1)) == ALL_IDS_NEWEST_FIRST[1:4]
    assert repo.list_with_truth(offset=16) == []


@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"limit": 1001}, {"offset": -1}, {"wrong": "true"}, {"wrong": 1}])
def test_bad_arguments_are_rejected(repo, kwargs):
    with pytest.raises(ValueError):
        repo.list_with_truth(**kwargs)


# ---------- stats: 손으로 계산한 값 ----------

def cells(s):
    return (s.tn, s.fp, s.fn, s.tp)


def test_stats_for_v1_all_ten(repo):
    s = repo.stats(model_version="v1")
    assert (s.total, s.ng_count) == (10, 5) and cells(s) == (3, 1, 2, 4) and s.n == 10
    assert s.ng_rate == 0.5 and s.accuracy == pytest.approx(0.7) and s.recall == pytest.approx(4 / 6)


@pytest.mark.parametrize("recent, expected_cells", [
    (1, (0, 0, 0, 1)),     # PRD-10 TP
    (3, (1, 0, 1, 1)),     # PRD-10 TP, 9 TN, 8 FN
    (5, (2, 0, 1, 2)),     # + 7 TN, 6 TP
    (10, (3, 1, 2, 4)),
    (100, (3, 1, 2, 4)),   # 전체보다 커도 있는 만큼만 센다
])
def test_stats_recent_window_of_v1(repo, recent, expected_cells):
    s = repo.stats(recent=recent, model_version="v1")
    assert cells(s) == expected_cells and s.n == min(recent, 10) and (s.total, s.ng_count) == (10, 5)


def test_stats_recent_5_of_v1_ratios(repo):
    s = repo.stats(recent=5, model_version="v1")
    assert s.accuracy == pytest.approx(4 / 5) and s.recall == pytest.approx(2 / 3)


def test_stats_without_actual_ng_has_no_recall(repo):
    s = repo.stats(model_version="v2")  # 정답이 모두 OK 인 검사
    assert (s.total, s.ng_count, s.ng_rate) == (4, 1, 0.25) and cells(s) == (3, 1, 0, 0)
    assert s.accuracy == 0.75 and s.recall is None


def test_stats_same_time_predictions_use_id_order(repo):
    one = repo.stats(recent=1, model_version="v3")  # 같은 시각 PRD-15(TP), PRD-16(FP): ID 가 큰 16 이 최신
    assert cells(one) == (0, 1, 0, 0) and one.accuracy == 0.0 and one.recall is None
    two = repo.stats(recent=2, model_version="v3")
    assert cells(two) == (0, 1, 0, 1) and two.accuracy == 0.5 and two.recall == 1.0


def test_stats_across_all_versions(repo):
    s = repo.stats()
    assert (s.total, s.ng_count) == (16, 8) and s.ng_rate == 0.5
    assert cells(s) == (6, 3, 2, 5) and s.accuracy == pytest.approx(11 / 16) and s.recall == pytest.approx(5 / 7)


def test_stats_recent_5_across_versions(repo):
    s = repo.stats(recent=5)  # PRD-16 FP, 15 TP, 14 TN, 13 TN, 12 FP
    assert cells(s) == (2, 2, 0, 1) and s.accuracy == pytest.approx(0.6) and s.recall == 1.0


@pytest.mark.parametrize("recent", [1, 3, 5, 10, 16, 100])
def test_stats_window_is_the_top_of_the_list(repo, recent):
    top = repo.list_with_truth(limit=min(recent, 16))
    manual = [0, 0, 0, 0]  # TN, FP, FN, TP
    for item in top:
        predicted_ng, true_ng = item.prediction.predicted_label.value == "NG", item.true_label.value == "NG"
        manual[(2 if true_ng else 0) + (1 if predicted_ng else 0)] += 1
    assert list(cells(repo.stats(recent=recent))) == manual


def test_stats_for_unknown_version_are_all_zero(repo):
    s = repo.stats(model_version="v9")
    assert (s.total, s.ng_count, s.n) == (0, 0, 0) and cells(s) == (0, 0, 0, 0)
    assert (s.ng_rate, s.accuracy, s.recall) == (None, None, None)


def test_stats_on_an_empty_database(session):
    s = PredictionRepository(session).stats()
    assert s == PredictionStats(total=0, ng_count=0, tn=0, fp=0, fn=0, tp=0) and s.ng_rate is None and s.accuracy is None


@pytest.mark.parametrize("recent", [0, 1001, -1, True, "5", 2.5, None])
def test_stats_rejects_bad_recent(repo, recent):
    with pytest.raises(ValueError):
        repo.stats(recent=recent)


def test_stats_properties_when_nothing_can_be_computed():
    assert PredictionStats(0, 0, 0, 0, 0, 0).accuracy is None
    assert PredictionStats(3, 1, 2, 0, 0, 0).recall is None  # 정답 NG 가 없다
    assert PredictionStats(3, 1, 2, 0, 0, 0).accuracy == 1.0
