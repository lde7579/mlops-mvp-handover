"""ModelVersionRepository: 등록, 오류 규칙(DuplicateError / ConflictError), production 조회, 목록."""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from backend.models import ModelVersionRow
from backend.repository import ConflictError, DuplicateError, ModelVersionRepository, Stage
from tests.backend.helpers import KST, at, make_metrics


@pytest.fixture
def repo(session):
    return ModelVersionRepository(session)


def test_register_fills_metrics_snapshot_and_created_at_from_the_same_object(repo, session):
    metrics = make_metrics("v1", train_data_snapshot={"train": 1500, "pool": 500, "seed": 7})
    record = repo.register(metrics, stage=Stage.PRODUCTION)
    row = session.scalars(select(ModelVersionRow)).one()
    assert row.metrics == metrics.to_dict()  # metrics.json 전체 내용
    assert row.train_data_snapshot == metrics.train_data_snapshot == {"train": 1500, "pool": 500, "seed": 7}
    assert row.metrics["train_data_snapshot"] == row.train_data_snapshot  # 중복 저장된 두 값이 일치
    assert row.created_at == metrics.created_at and record.created_at == metrics.created_at
    assert row.stage == "production" and (row.approved_by, row.approved_at) == (None, None)


def test_read_back_gives_the_same_model_metrics(repo):
    metrics = make_metrics("v3", created_minutes=5)
    repo.register(metrics)
    got = repo.get("v3")
    assert got.metrics == metrics and got.version == "v3" and got.stage is Stage.CANDIDATE
    assert got.train_data_snapshot == {"train": 2000, "pool": 0, "seed": 42}
    assert got.created_at.utcoffset() == timedelta(0) and got.created_at == at(5)
    assert repo.get("v9") is None


def test_default_stage_is_candidate_and_approval_fields_are_stored_in_utc(repo):
    assert repo.register(make_metrics("v1")).stage is Stage.CANDIDATE
    approved = repo.register(make_metrics("v2", created_minutes=1), stage="staging", approved_by="admin",
                             approved_at=datetime(2026, 10, 18, 12, 0, tzinfo=KST))
    got = repo.get("v2")
    assert got.stage is Stage.STAGING and got.approved_by == "admin"
    assert got.approved_at == datetime(2026, 10, 18, 3, 0, tzinfo=timezone.utc) == approved.approved_at


def test_duplicate_version_raises_duplicate_error_and_keeps_the_first(repo):
    repo.register(make_metrics("v1"), stage=Stage.PRODUCTION)
    with pytest.raises(DuplicateError):
        repo.register(make_metrics("v1", threshold=0.9), stage=Stage.CANDIDATE)
    assert len(repo.list()) == 1 and repo.get("v1").stage is Stage.PRODUCTION and repo.get("v1").metrics.threshold == 0.5


def test_second_production_raises_conflict_error_and_nothing_is_added(repo):
    repo.register(make_metrics("v1"), stage=Stage.PRODUCTION)
    with pytest.raises(ConflictError, match="v1"):
        repo.register(make_metrics("v2", created_minutes=1), stage=Stage.PRODUCTION)
    assert [r.version for r in repo.list()] == ["v1"]
    assert repo.get_production().version == "v1"


def test_many_non_production_versions_are_allowed(repo):
    repo.register(make_metrics("v1"), stage=Stage.PRODUCTION)
    for i, stage in enumerate((Stage.ARCHIVED, Stage.ARCHIVED, Stage.STAGING, Stage.CANDIDATE, Stage.REJECTED), start=2):
        repo.register(make_metrics(f"v{i}", created_minutes=i), stage=stage)
    assert len(repo.list()) == 6 and repo.get_production().version == "v1"


def test_get_production(repo):
    assert repo.get_production() is None
    repo.register(make_metrics("v1"), stage=Stage.CANDIDATE)
    assert repo.get_production() is None
    repo.register(make_metrics("v2", created_minutes=1), stage=Stage.PRODUCTION)
    assert repo.get_production().version == "v2"


def test_list_is_oldest_first_and_ties_by_version(repo):
    repo.register(make_metrics("v2", created_minutes=5))
    repo.register(make_metrics("v1", created_minutes=1))
    repo.register(make_metrics("v3", created_minutes=5))
    assert [r.version for r in repo.list()] == ["v1", "v2", "v3"]


@pytest.mark.parametrize("stage", ["approved", "", None, 3])
def test_unknown_stage_is_rejected(repo, stage):
    with pytest.raises(ValueError):
        repo.register(make_metrics("v1"), stage=stage)
    assert repo.list() == []


def test_wrong_argument_types_are_rejected(repo):
    with pytest.raises(TypeError):
        repo.register(make_metrics("v1").to_dict())
    with pytest.raises(ValueError):
        repo.register(make_metrics("v1"), approved_by=5)
    assert repo.list() == []


def test_times_without_timezone_are_rejected(repo):
    naive_created = make_metrics("v1", created_at="2026-10-18T09:00:00")  # 스키마는 허용하지만 저장소는 거부
    with pytest.raises(ValueError, match="시간대"):
        repo.register(naive_created)
    with pytest.raises(ValueError, match="시간대"):
        repo.register(make_metrics("v2"), approved_at=datetime(2026, 10, 18, 12, 0))
    assert repo.list() == []
