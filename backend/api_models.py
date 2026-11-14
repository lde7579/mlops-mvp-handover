"""API 요청·응답 모델 (pydantic). 계약 스키마(common/schemas.py)와 별개로, 화면과 주고받는 JSON 형식이다."""
from datetime import datetime, timezone
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

from backend.repository import (
    ApprovalRecord,
    ApprovalStatus,
    AuditAction,
    AuditLogRecord,
    DeploymentAction,
    DeploymentRecord,
    ModelVersionRecord,
    PredictionStats,
    PredictionWithTruth,
    RunStatus,
    RunStep,
    RunTrigger,
    Stage,
    TrainingRunRecord,
)
from backend.monitoring import ACCURACY_THRESHOLD, CRITERIA_CONFIRMED, SHIFT_NOTE, DriftResult, PerformanceResult
from backend.security import Role
from backend.simulator_runner import DEFAULT_DEFECT_RATE, DEFAULT_INTERVAL_SECONDS, SimulatorStatus
from common.schemas import DataSplit, Inspection, Label, Prediction, Scenario


class RequestModel(BaseModel):
    """요청 본문의 공통 규칙: 문서에 없는 항목이 오면 400 (오타를 조용히 무시하지 않는다)."""
    model_config = ConfigDict(extra="forbid")


class ErrorResponse(BaseModel):
    error: str = Field(description="오류 이름: bad_request, unauthorized, forbidden, not_found, method_not_allowed, conflict, "
                                   "server_error, service_unavailable")
    message: str


class LoginRequest(RequestModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class LoginResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(description="토큰 유효 시간(초)")
    role: Role
    username: str


class GenerateRequest(RequestModel):
    scenario: Scenario = Field(default=Scenario.NORMAL, description="normal 또는 drift")


class StartRequest(RequestModel):
    """POST /simulator/start (A23). 숫자만 받는다 (문자열 "0.5" 나 true 는 400)."""
    defect_rate: Annotated[float, Field(strict=True, ge=0.0, le=1.0, description="불량률 0.0~1.0")] = DEFAULT_DEFECT_RATE
    interval_seconds: Annotated[float, Field(strict=True, ge=1.0, le=3.0, description="생성 간격(초) 1.0~3.0")] = DEFAULT_INTERVAL_SECONDS


class DriftRequest(RequestModel):
    """POST /simulator/drift (A23). enabled 가 true 이면 drift_type 필수, false 이면 drift_type 금지 (라우터에서 확인)."""
    enabled: StrictBool
    drift_type: Optional[Literal["dark", "noise", "shift"]] = None


class DriftState(BaseModel):
    enabled: bool
    drift_type: Optional[str]


class SimulatorStatusOut(BaseModel):
    """자동 생성 상태 9개 필드 (A23). 실행 중이 아니면 interval_seconds·defect_rate·started_at 이 null."""
    running: bool
    interval_seconds: Optional[float]
    defect_rate: Optional[float]
    drift: DriftState
    generated: int
    judged: int
    judge_failures: int
    last_error: Optional[str]
    started_at: Optional[datetime]

    @classmethod
    def of(cls, s: SimulatorStatus) -> "SimulatorStatusOut":
        return cls(running=s.running, interval_seconds=s.interval_seconds, defect_rate=s.defect_rate,
                   drift=DriftState(enabled=s.drift_enabled, drift_type=s.drift_type), generated=s.generated,
                   judged=s.judged, judge_failures=s.judge_failures, last_error=s.last_error,
                   started_at=None if s.started_at is None else s.started_at.astimezone(timezone.utc))


class CriteriaOut(BaseModel):
    """성능 저하 기준. 5주차 측정으로 확정한 값이고 confirmed 는 true (requirements.md D-3, api-v1.md A27)."""
    recent_n: int = Field(description="최근 N건의 N")
    accuracy_threshold: float = Field(description="이 값보다 낮으면 성능 저하")
    confirmed: bool = Field(description="5주차 측정으로 확정된 값인가")


class PerformanceRecentOut(BaseModel):
    """RecentStats 와 같은 값에, 센 판정의 모델 버전을 더한 것."""
    model_version: Optional[str] = Field(description="센 판정의 모델 버전 (생략하면 현재 production, 없으면 null)")
    n: int = Field(description="실제로 센 건수 (최대 recent)")
    accuracy: Optional[float] = Field(description="(TN+TP)/n. n이 0이면 null")
    recall: Optional[float] = Field(description="TP/(TP+FN). 정답이 NG인 건이 없으면 null")
    confusion_matrix: list[list[int]] = Field(description="[[TN, FP], [FN, TP]], 양성은 NG, 정답은 검사의 true_label")


class PerformancePointOut(BaseModel):
    bucket_start: datetime = Field(description="구간 시작 (UTC, predicted_at 기준)")
    model_version: str
    n: int
    accuracy: float
    recall: Optional[float] = Field(description="구간 안에 정답 NG가 없으면 null")


class PerformanceOut(BaseModel):
    since: datetime
    until: datetime
    bucket_seconds: int
    criteria: CriteriaOut
    recent: PerformanceRecentOut
    points: list[PerformancePointOut]

    @classmethod
    def of(cls, r: "PerformanceResult") -> "PerformanceOut":
        return cls(
            since=r.since, until=r.until, bucket_seconds=r.bucket_seconds,
            criteria=CriteriaOut(recent_n=r.recent_n, accuracy_threshold=ACCURACY_THRESHOLD, confirmed=CRITERIA_CONFIRMED),
            recent=PerformanceRecentOut(model_version=r.recent_version, n=r.recent.n, accuracy=r.recent.accuracy, recall=r.recent.recall,
                                        confusion_matrix=[[r.recent.tn, r.recent.fp], [r.recent.fn, r.recent.tp]]),
            points=[PerformancePointOut(bucket_start=p.bucket_start, model_version=p.model_version, n=p.n, accuracy=p.accuracy,
                                        recall=p.recall) for p in r.points])


class PerformanceWarningOut(BaseModel):
    model_version: Optional[str]
    active: Optional[bool] = Field(description="최신 N건 정확도가 임계값 미만이면 true. N건 미만이거나 production이 없으면 null(판단 보류)")
    n: int
    accuracy: Optional[float]
    recent_n: int
    accuracy_threshold: float
    confirmed: bool


class MeanStd(BaseModel):
    mean: float
    std: float


class BaselineOut(BaseModel):
    source: str = Field(description="기준을 계산한 폴더")
    n: int = Field(description="계산에 쓴 이미지 수")
    brightness: MeanStd
    noise: MeanStd


class CurrentOut(BaseModel):
    brightness: float
    noise: float


class DistanceOut(BaseModel):
    brightness_sigma: Optional[float]
    noise_sigma: Optional[float]


class InputDistributionOut(BaseModel):
    window_n: int = Field(description="current 를 계산한 이미지 수")
    baseline: Optional[BaselineOut]
    current: Optional[CurrentOut]
    distance: Optional[DistanceOut]
    note: str


class DriftOut(BaseModel):
    performance_warning: PerformanceWarningOut
    input_distribution: InputDistributionOut

    @classmethod
    def of(cls, r: "DriftResult") -> "DriftOut":
        b = r.baseline
        return cls(
            performance_warning=PerformanceWarningOut(
                model_version=r.model_version, active=r.active, n=r.n, accuracy=r.accuracy, recent_n=r.recent_n,
                accuracy_threshold=ACCURACY_THRESHOLD, confirmed=CRITERIA_CONFIRMED),
            input_distribution=InputDistributionOut(
                window_n=r.window_n,
                baseline=None if b is None else BaselineOut(
                    source=b.source, n=b.n, brightness=MeanStd(mean=b.brightness_mean, std=b.brightness_std),
                    noise=MeanStd(mean=b.noise_mean, std=b.noise_std)),
                current=None if r.current is None else CurrentOut(brightness=r.current[0], noise=r.current[1]),
                distance=None if r.distance is None else DistanceOut(brightness_sigma=r.distance[0], noise_sigma=r.distance[1]),
                note=SHIFT_NOTE))


class InspectionOut(BaseModel):
    """계약 2장 inspection 8개 필드."""
    inspection_id: str
    captured_at: datetime
    image_path: str
    true_label: Label
    missing_parts: list[int]
    scenario: Scenario
    data_split: DataSplit
    drift_type: Optional[str]

    @classmethod
    def of(cls, i: Inspection) -> "InspectionOut":
        # 시각은 항상 UTC 로 내보낸다 (방금 만든 검사와 DB 에서 읽은 검사의 표기가 같아야 한다)
        return cls(inspection_id=i.inspection_id, captured_at=i.captured_at.astimezone(timezone.utc), image_path=i.image_path,
                   true_label=i.true_label, missing_parts=list(i.missing_parts), scenario=i.scenario,
                   data_split=i.data_split, drift_type=i.drift_type)


class PredictionOut(BaseModel):
    """계약 3장 prediction 8개 필드."""
    prediction_id: str
    inspection_id: str
    model_version: str
    predicted_label: Label
    score: float
    threshold: float
    latency_ms: int
    predicted_at: datetime

    @classmethod
    def of(cls, p: Prediction) -> "PredictionOut":
        return cls(prediction_id=p.prediction_id, inspection_id=p.inspection_id, model_version=p.model_version,
                   predicted_label=p.predicted_label, score=p.score, threshold=p.threshold,
                   latency_ms=p.latency_ms, predicted_at=p.predicted_at.astimezone(timezone.utc))


class PredictionWithTruthOut(PredictionOut):
    """GET /predictions 의 항목: prediction 8개 필드 + 검사의 정답과 정오 (정오는 저장하지 않고 조회 때 계산)."""
    true_label: Label
    correct: bool = Field(description="predicted_label 과 true_label 이 같은가")

    @classmethod
    def with_truth(cls, item: PredictionWithTruth) -> "PredictionWithTruthOut":
        return cls(**PredictionOut.of(item.prediction).model_dump(), true_label=item.true_label, correct=item.correct)


class PredictionList(BaseModel):
    items: list[PredictionWithTruthOut]
    total: int = Field(description="필터를 적용한 전체 건수 (limit·offset과 무관)")


class RecentStats(BaseModel):
    n: int = Field(description="실제로 센 건수 (최대 recent)")
    accuracy: Optional[float] = Field(description="(TN+TP)/n. n이 0이면 null")
    recall: Optional[float] = Field(description="TP/(TP+FN). 정답이 NG인 건이 없으면 null")
    confusion_matrix: list[list[int]] = Field(description="[[TN, FP], [FN, TP]], 양성은 NG, 정답은 검사의 true_label")


class PredictionStatsOut(BaseModel):
    total: int = Field(description="판정 전체 건수")
    ng_count: int = Field(description="판정이 NG인 건수")
    ng_rate: Optional[float] = Field(description="ng_count/total (판정 기준). total이 0이면 null")
    recent: RecentStats

    @classmethod
    def of(cls, s: PredictionStats) -> "PredictionStatsOut":
        return cls(total=s.total, ng_count=s.ng_count, ng_rate=s.ng_rate,
                   recent=RecentStats(n=s.n, accuracy=s.accuracy, recall=s.recall,
                                      confusion_matrix=[[s.tn, s.fp], [s.fn, s.tp]]))


class ClassMetricsOut(BaseModel):
    accuracy: float
    recall: float
    precision: float
    f1: float
    confusion_matrix: list[list[int]]
    n: int


class ModelMetricsOut(BaseModel):
    """metrics.json 내용. 시각(created_at)만 UTC Z 로 바꿔 내보내므로 파일에 적힌 표기와 다를 수 있다."""
    version: str
    threshold: float
    normal: ClassMetricsOut
    drift: ClassMetricsOut
    train_data_snapshot: dict[str, int]
    created_at: datetime


class ModelVersionOut(BaseModel):
    """계약 5장 model_version 7개 필드. 시각(created_at, approved_at, metrics.created_at)은 UTC Z."""
    version: str
    stage: Stage
    metrics: ModelMetricsOut
    train_data_snapshot: dict[str, int]
    created_at: datetime
    approved_by: Optional[str]
    approved_at: Optional[datetime]

    @classmethod
    def of(cls, r: ModelVersionRecord) -> "ModelVersionOut":
        m = r.metrics.to_dict()
        m["created_at"] = r.metrics.created_at.astimezone(timezone.utc)
        return cls(version=r.version, stage=r.stage, metrics=ModelMetricsOut(**m),
                   train_data_snapshot=dict(r.train_data_snapshot), created_at=r.created_at.astimezone(timezone.utc),
                   approved_by=r.approved_by, approved_at=r.approved_at.astimezone(timezone.utc) if r.approved_at else None)


class ModelList(BaseModel):
    items: list[ModelVersionOut]
    total: int


class PredictRequest(RequestModel):
    inspection_id: str = Field(min_length=1, max_length=64, description="판정할 검사 ID")


class InspectionDetail(InspectionOut):
    predictions: list[PredictionOut] = Field(description="이 검사의 판정 이력 (최신 먼저). 판정 전이면 빈 목록")


class InspectionList(BaseModel):
    items: list[InspectionOut]
    total: int = Field(description="필터를 적용한 전체 건수 (limit·offset과 무관)")


class ModelStatus(BaseModel):
    production_version: Optional[str] = Field(description="DB 에 production 으로 등록된 모델 버전 (없으면 null)")
    loaded: bool = Field(description="추론부가 production_version 을 실제로 불러와 판정에 쓸 수 있는가 "
                                     "(운영 모델이 없거나, 아직 불러오지 않았거나, 불러오기에 실패했으면 false)")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: Literal["ok", "error"]
    model: ModelStatus


class TrainingRunRequest(RequestModel):
    """4-B 에서는 manual 만 받는다 (api-v1.md A16). 본문이 없거나 trigger 를 생략해도 manual."""
    trigger: Literal["manual"] = "manual"


class TrainingRunOut(BaseModel):
    """계약 7장 training_run 9개 필드 + train_data_snapshot(결과 버전 모델의 값, 없으면 null). 시각은 UTC Z (api-v1.md A15, A16)."""
    run_id: str
    status: RunStatus
    trigger: RunTrigger
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    result_version: Optional[str]
    step: Optional[RunStep]
    error_message: Optional[str]
    gate_result: Optional[dict]
    train_data_snapshot: Optional[dict[str, int]]

    @classmethod
    def of(cls, r: TrainingRunRecord, snapshot: Optional[dict] = None) -> "TrainingRunOut":
        def utc(value):
            return None if value is None else value.astimezone(timezone.utc)

        return cls(run_id=r.run_id, status=r.status, trigger=r.trigger, started_at=utc(r.started_at), finished_at=utc(r.finished_at),
                   result_version=r.result_version, step=r.step, error_message=r.error_message, gate_result=r.gate_result,
                   train_data_snapshot=None if snapshot is None else dict(snapshot))


class TrainingRunList(BaseModel):
    items: list[TrainingRunOut]
    total: int = Field(description="필터를 적용한 전체 건수 (limit·offset과 무관)")


# ---------- 승인·배포·감사 (api-v1.md A17~A22) ----------

def _utc(value):
    return None if value is None else value.astimezone(timezone.utc)


class ApprovalOut(BaseModel):
    """계약 6장 approval_request 8개 필드. 시각은 UTC Z, 처리 전에는 reason·handled_by·handled_at 이 null."""
    approval_id: str
    model_version: str
    requested_by: str
    status: ApprovalStatus
    reason: Optional[str]
    handled_by: Optional[str]
    handled_at: Optional[datetime]
    created_at: datetime

    @classmethod
    def of(cls, a: ApprovalRecord) -> "ApprovalOut":
        return cls(approval_id=a.approval_id, model_version=a.model_version, requested_by=a.requested_by, status=a.status,
                   reason=a.reason, handled_by=a.handled_by, handled_at=_utc(a.handled_at), created_at=_utc(a.created_at))


class ApprovalList(BaseModel):
    items: list[ApprovalOut]
    total: int = Field(description="필터를 적용한 전체 건수 (limit·offset과 무관)")


class DeploymentOut(BaseModel):
    """계약 7장 deployment 6개 필드. 시각은 UTC Z, 사유를 안 냈으면 reason 은 null."""
    deployment_id: str
    version: str
    action: DeploymentAction
    reason: Optional[str]
    actor: str
    at: datetime

    @classmethod
    def of(cls, d: DeploymentRecord) -> "DeploymentOut":
        return cls(deployment_id=d.deployment_id, version=d.version, action=d.action, reason=d.reason, actor=d.actor, at=_utc(d.at))


class DeploymentList(BaseModel):
    items: list[DeploymentOut]
    total: int = Field(description="필터를 적용한 전체 건수 (limit·offset과 무관)")


class AuditLogOut(BaseModel):
    """계약 7장 audit_log 6개 필드. 시각은 UTC Z."""
    log_id: str
    actor: str
    role: str
    action: AuditAction
    target: str
    at: datetime

    @classmethod
    def of(cls, a: AuditLogRecord) -> "AuditLogOut":
        return cls(log_id=a.log_id, actor=a.actor, role=a.role.value, action=a.action, target=a.target, at=_utc(a.at))


class AuditLogList(BaseModel):
    items: list[AuditLogOut]
    total: int = Field(description="필터를 적용한 전체 건수 (limit·offset과 무관)")


def _not_blank(value):
    if value is not None and not value.strip():
        raise ValueError("공백만으로 된 값은 안 됩니다")
    return value


class ApproveRequest(RequestModel):
    """승인 요청. 사유는 선택이지만, 보냈다면 공백만으로 된 값은 안 된다."""
    reason: Optional[str] = Field(default=None, min_length=1, max_length=500)

    @field_validator("reason")
    @classmethod
    def check_reason(cls, value):
        return _not_blank(value)


class ReasonRequest(RequestModel):
    """거절·롤백 요청. 사유는 필수이고 공백만으로 된 값은 안 된다 (e2e X5)."""
    reason: str = Field(min_length=1, max_length=500)

    @field_validator("reason")
    @classmethod
    def check_reason(cls, value):
        return _not_blank(value)


class ReloadOut(BaseModel):
    loaded_version: str = Field(description="교체가 끝난 뒤 추론부가 쓰는 모델 버전")
    elapsed_ms: int = Field(description="새 모델을 읽는 데 걸린 시간(밀리초)")


class ApproveOut(BaseModel):
    approval: ApprovalOut
    deployment: DeploymentOut
    new_version: str
    old_version: Optional[str] = Field(description="내려간 이전 운영 버전 (없었으면 null)")
    reload: ReloadOut


class RejectOut(BaseModel):
    approval: ApprovalOut
    version: str
    stage: Literal["rejected"]


class RollbackOut(BaseModel):
    deployment: DeploymentOut
    new_version: str
    old_version: Optional[str] = Field(description="내려간 이전 운영 버전 (없었으면 null)")
    reload: ReloadOut
