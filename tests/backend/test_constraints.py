"""DB 제약 확인: 저장소·스키마를 거치지 않고 원시 SQL 로 넣어서 DB 자체가 막는지 본다."""
import pytest
from sqlalchemy.exc import IntegrityError

from tests.backend.helpers import raw_inspection, raw_model_version, raw_prediction


@pytest.fixture
def conn(engine):
    with engine.begin() as c:
        yield c


def test_valid_rows_are_accepted(conn):
    raw_model_version(conn)
    raw_inspection(conn)
    raw_prediction(conn)


@pytest.mark.parametrize("over", [{"label": "ok"}, {"label": "BAD"}, {"label": ""}, {"scenario": "Normal"},
                                  {"scenario": "other"}, {"split": "validation"}, {"split": "test"}])
def test_inspection_value_sets_are_enforced(conn, over):
    with pytest.raises(IntegrityError, match="CHECK"):
        raw_inspection(conn, **over)


def test_inspection_allows_any_drift_type_text_and_empty(conn):
    raw_inspection(conn, id="INS-000001", drift="blur", scenario="drift")  # 계약: dark / noise / shift "등"
    raw_inspection(conn, id="INS-000002", drift=None)


def test_inspection_primary_key_is_unique(conn):
    raw_inspection(conn)
    with pytest.raises(IntegrityError, match="UNIQUE"):
        raw_inspection(conn)


@pytest.mark.parametrize("stage", ["approved", "Production", "", "new"])
def test_stage_value_set_is_enforced(conn, stage):
    with pytest.raises(IntegrityError, match="CHECK"):
        raw_model_version(conn, stage=stage)


@pytest.mark.parametrize("stage", ["candidate", "staging", "production", "archived", "rejected"])
def test_all_five_stages_are_accepted(conn, stage):
    raw_model_version(conn, stage=stage)


def test_only_one_production_but_many_other_stages(conn):
    raw_model_version(conn, version="v1", stage="production")
    for i, stage in enumerate(("archived", "archived", "staging", "candidate", "rejected"), start=2):
        raw_model_version(conn, version=f"v{i}", stage=stage)
    with pytest.raises(IntegrityError, match="UNIQUE"):
        raw_model_version(conn, version="v9", stage="production")


def test_production_must_be_archived_before_next_is_promoted(engine):
    """4주차 승인 순서: 기존 production 을 archived 로 바꾼 뒤 새 버전을 production 으로 바꿔야 한다."""
    with engine.begin() as c:
        raw_model_version(c, version="v1", stage="production")
        raw_model_version(c, version="v2", stage="staging")
    with pytest.raises(IntegrityError, match="UNIQUE"):  # 순서가 반대면 유니크 위반
        with engine.begin() as c:
            c.exec_driver_sql("UPDATE model_version SET stage = 'production' WHERE version = 'v2'")
    with engine.begin() as c:  # 올바른 순서
        c.exec_driver_sql("UPDATE model_version SET stage = 'archived' WHERE version = 'v1'")
        c.exec_driver_sql("UPDATE model_version SET stage = 'production' WHERE version = 'v2'")
        assert c.exec_driver_sql("SELECT version FROM model_version WHERE stage = 'production'").scalar() == "v2"


def test_approved_fields_may_be_empty(conn):
    raw_model_version(conn, version="v1")
    raw_model_version(conn, version="v2", by="admin", at="2026-10-18T01:00:00.000000+00:00")


@pytest.mark.parametrize("score", [-0.01, 1.01, 2, -1])
def test_prediction_score_range_is_enforced(conn, score):
    raw_model_version(conn)
    raw_inspection(conn)
    with pytest.raises(IntegrityError, match="CHECK"):
        raw_prediction(conn, score=score)


@pytest.mark.parametrize("score", [0.0, 1.0])
def test_prediction_score_bounds_are_accepted(conn, score):
    raw_model_version(conn)
    raw_inspection(conn)
    raw_prediction(conn, score=score, label="NG" if score >= 0.5 else "OK")


def test_prediction_label_value_set_is_enforced(conn):
    raw_model_version(conn)
    raw_inspection(conn)
    with pytest.raises(IntegrityError, match="CHECK"):
        raw_prediction(conn, label="maybe")


def test_foreign_keys_reject_unknown_inspection_and_model_version(conn):
    raw_model_version(conn)
    raw_inspection(conn)
    with pytest.raises(IntegrityError, match="FOREIGN KEY"):
        raw_prediction(conn, ins="INS-999999")
    with pytest.raises(IntegrityError, match="FOREIGN KEY"):
        raw_prediction(conn, ver="v9")


def test_prediction_primary_key_is_unique(conn):
    raw_model_version(conn)
    raw_inspection(conn)
    raw_prediction(conn)
    with pytest.raises(IntegrityError, match="UNIQUE"):
        raw_prediction(conn)


def test_one_inspection_can_have_predictions_from_several_models(conn):
    raw_model_version(conn, version="v1", stage="archived")
    raw_model_version(conn, version="v2", stage="production")
    raw_inspection(conn)
    raw_prediction(conn, id="PRD-000001", ver="v1")
    raw_prediction(conn, id="PRD-000002", ver="v2")


def test_label_score_rule_is_not_duplicated_in_the_db(conn):
    # 규칙(score >= threshold 이면 NG)은 스키마가 검사한다. DB 는 같은 규칙을 한 번 더 두지 않는다.
    # (읽을 때 걸러지는지는 test_prediction_repository 의 읽기 검증 테스트가 확인한다)
    raw_model_version(conn)
    raw_inspection(conn)
    raw_prediction(conn, score=0.1, threshold=0.5, label="NG")
