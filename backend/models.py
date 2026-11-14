"""DB 테이블 7개 (docs/data-contract.md v1.4): inspection(2장), prediction(3장), model_version(5장),
approval_request(6장), training_run·deployment·audit_log(7장).

컬럼 이름은 계약 필드와 같다. 계약에 없는 컬럼은 없다 (training_run 의 step, error_message, gate_result 는 C7 로 계약에 들어왔다).
- 값 집합이 정해진 필드(라벨, scenario, data_split, stage)는 문자열 + CHECK 제약이다.
- drift_type 은 계약이 "dark / noise / shift 등"이라 값을 제한하지 않는다.
- 정오 여부는 계약대로 저장하지 않는다 (조회 때 predicted_label 과 true_label 을 비교).
- predicted_label 과 score·threshold 의 일치(score >= threshold 이면 NG)는 스키마(common/schemas.py)가 저장·읽기에서
  검사한다. DB 에 같은 규칙을 한 번 더 두지 않는다 (규칙이 바뀔 때 두 곳을 고치지 않기 위해).
- 모델 파일 위치는 컬럼이 아니다: 계약 4장의 models/<version>/ 규칙으로 계산한다.
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, CheckConstraint, Float, ForeignKey, Index, Integer, String, text
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base, UtcDateTime


class InspectionRow(Base):
    __tablename__ = "inspection"

    inspection_id: Mapped[str] = mapped_column(String, primary_key=True)
    captured_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    image_path: Mapped[str] = mapped_column(String, nullable=False)  # 프로젝트 루트 기준 상대경로
    true_label: Mapped[str] = mapped_column(String, nullable=False)
    missing_parts: Mapped[list] = mapped_column(JSON, nullable=False)  # list[int]
    scenario: Mapped[str] = mapped_column(String, nullable=False)
    drift_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    data_split: Mapped[str] = mapped_column(String, nullable=False)

    __table_args__ = (
        CheckConstraint("true_label IN ('OK', 'NG')", name="ck_inspection_true_label"),
        CheckConstraint("scenario IN ('normal', 'drift')", name="ck_inspection_scenario"),
        CheckConstraint("data_split IN ('train', 'pool', 'test_normal', 'test_drift')", name="ck_inspection_data_split"),
        Index("ix_inspection_data_split", "data_split"),
        Index("ix_inspection_captured_at", "captured_at"),
    )


class ModelVersionRow(Base):
    __tablename__ = "model_version"

    version: Mapped[str] = mapped_column(String, primary_key=True)
    stage: Mapped[str] = mapped_column(String, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)  # metrics.json 전체 내용
    train_data_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    approved_by: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    approved_at: Mapped[Optional[datetime]] = mapped_column(UtcDateTime, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "stage IN ('candidate', 'staging', 'production', 'archived', 'rejected')", name="ck_model_version_stage"),
        # production 은 최대 1개 (계약 5장: 승인 시 기존 production 은 archived)
        Index("uq_model_version_one_production", "stage", unique=True, sqlite_where=text("stage = 'production'")),
    )


class PredictionRow(Base):
    __tablename__ = "prediction"

    prediction_id: Mapped[str] = mapped_column(String, primary_key=True)
    inspection_id: Mapped[str] = mapped_column(ForeignKey("inspection.inspection_id"), nullable=False)
    model_version: Mapped[str] = mapped_column(ForeignKey("model_version.version"), nullable=False)
    predicted_label: Mapped[str] = mapped_column(String, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)

    __table_args__ = (
        CheckConstraint("predicted_label IN ('OK', 'NG')", name="ck_prediction_predicted_label"),
        CheckConstraint("score >= 0 AND score <= 1", name="ck_prediction_score"),
        Index("ix_prediction_inspection_id", "inspection_id"),
        Index("ix_prediction_model_version", "model_version"),
        Index("ix_prediction_predicted_at", "predicted_at"),
    )


class ApprovalRequestRow(Base):
    __tablename__ = "approval_request"

    approval_id: Mapped[str] = mapped_column(String, primary_key=True)
    model_version: Mapped[str] = mapped_column(ForeignKey("model_version.version"), nullable=False)
    requested_by: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    handled_by: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    handled_at: Mapped[Optional[datetime]] = mapped_column(UtcDateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)

    __table_args__ = (
        CheckConstraint("status IN ('pending', 'approved', 'rejected')", name="ck_approval_request_status"),
        # 같은 버전에 pending 은 1개만 (계약 6장 C7)
        Index("uq_approval_one_pending_per_version", "model_version", unique=True, sqlite_where=text("status = 'pending'")),
        Index("ix_approval_request_status", "status"),
    )


class TrainingRunRow(Base):
    __tablename__ = "training_run"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    trigger: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(UtcDateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(UtcDateTime, nullable=True)
    result_version: Mapped[Optional[str]] = mapped_column(ForeignKey("model_version.version"), nullable=True)
    step: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    gate_result: Mapped[Optional[dict]] = mapped_column(JSON(none_as_null=True), nullable=True)  # 비어 있으면 SQL NULL

    __table_args__ = (
        CheckConstraint("status IN ('queued', 'running', 'succeeded', 'failed')", name="ck_training_run_status"),
        CheckConstraint("trigger IN ('manual', 'count', 'performance')", name="ck_training_run_trigger"),
        CheckConstraint("step IS NULL OR step IN ('data', 'train', 'evaluate', 'gate')", name="ck_training_run_step"),
    )


class DeploymentRow(Base):
    __tablename__ = "deployment"

    deployment_id: Mapped[str] = mapped_column(String, primary_key=True)
    version: Mapped[str] = mapped_column(ForeignKey("model_version.version"), nullable=False)
    action: Mapped[str] = mapped_column(String, nullable=False)
    reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    actor: Mapped[str] = mapped_column(String, nullable=False)
    at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)

    __table_args__ = (
        CheckConstraint("action IN ('deploy', 'rollback')", name="ck_deployment_action"),
        Index("ix_deployment_at", "at"),
    )


class AuditLogRow(Base):
    __tablename__ = "audit_log"

    log_id: Mapped[str] = mapped_column(String, primary_key=True)
    actor: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    action: Mapped[str] = mapped_column(String, nullable=False)
    target: Mapped[str] = mapped_column(String, nullable=False)
    at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)

    __table_args__ = (
        Index("ix_audit_log_at", "at"),
    )
