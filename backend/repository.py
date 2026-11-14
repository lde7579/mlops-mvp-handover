"""저장소 계층: DB 행과 계약 스키마(common/schemas.py)를 오가며 저장·조회한다.

- 이미지는 폴더, 메타데이터는 DB다. 이 계층은 파일을 읽거나 쓰지 않고, image_path(상대경로)를 문자열로만 저장한다.
- 도메인 객체는 Inspection, Prediction, ModelMetrics 다. 저장할 때 스키마 검증을 통과해야 하고, 읽을 때도 다시 검증한다
  (DB 가 손상돼도 계약 위반 데이터가 코드로 올라오지 않는다). SQLAlchemy 행 객체는 밖으로 내보내지 않는다.
- 저장소는 commit 하지 않는다. 호출하는 쪽의 session_scope() 가 커밋/롤백한다.
- 시각은 시간대가 있어야 한다 (UTC 로 저장, 읽으면 UTC 시각).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Optional

from sqlalchemy import and_, case, func, select, text, update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from backend.database import ensure_aware, to_utc_text
from backend.models import (
    ApprovalRequestRow,
    AuditLogRow,
    DeploymentRow,
    InspectionRow,
    ModelVersionRow,
    PredictionRow,
    TrainingRunRow,
)
from common.schemas import DataSplit, InferenceOutput, Inspection, Label, ModelMetrics, Prediction, Scenario

DEFAULT_LIMIT = 50
MAX_LIMIT = 1000
DEFAULT_RECENT = 100  # stats 의 최근 N건 기본값
PREDICTION_ID_PREFIX = "PRD-"  # 계약 1장 권장 형식: PRD-000001
PREDICTION_ID_WIDTH = 6
ID_ATTEMPTS = 3


class DuplicateError(Exception):
    """이미 있는 ID(또는 버전)다."""


class NotFoundError(Exception):
    """참조하는 대상이 없다."""


class ConflictError(Exception):
    """제약과 충돌한다 (예: production 이 이미 있는데 production 으로 등록)."""


class InvalidTransitionError(ConflictError):
    """지금 상태에서는 할 수 없는 전이다 (허용되지 않은 stage 변경, 이미 처리된 승인 요청 등). API 에서는 409 가 된다."""


class Stage(StrEnum):
    CANDIDATE = "candidate"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    REJECTED = "rejected"


# 계약 5장 상태 전이. production -> archived 는 승격·롤백 안에서만 쓰므로 여기에 넣지 않는다 (archive_production).
ALLOWED_TRANSITIONS = frozenset({
    (Stage.CANDIDATE, Stage.STAGING),     # 게이트 통과
    (Stage.CANDIDATE, Stage.REJECTED),    # 게이트 미통과
    (Stage.STAGING, Stage.PRODUCTION),    # 관리자 승인
    (Stage.STAGING, Stage.REJECTED),      # 관리자 거절
    (Stage.ARCHIVED, Stage.PRODUCTION),   # 관리자 롤백
})


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class RunTrigger(StrEnum):
    MANUAL = "manual"
    COUNT = "count"
    PERFORMANCE = "performance"


class RunStep(StrEnum):
    DATA = "data"
    TRAIN = "train"
    EVALUATE = "evaluate"
    GATE = "gate"


class DeploymentAction(StrEnum):
    DEPLOY = "deploy"
    ROLLBACK = "rollback"


class AuditAction(StrEnum):
    GATE_PASS = "gate_pass"
    GATE_FAIL = "gate_fail"
    APPROVE = "approve"
    REJECT = "reject"
    ROLLBACK = "rollback"
    APPROVE_FAIL = "approve_fail"    # 새 모델을 불러오지 못해 승인하지 않았다 (계약 C9)
    ROLLBACK_FAIL = "rollback_fail"  # 새 모델을 불러오지 못해 롤백하지 않았다 (계약 C9)


class Role(StrEnum):
    OPERATOR = "operator"
    ADMIN = "admin"
    SYSTEM = "system"


@dataclass
class PredictionWithTruth:
    """판정 한 건과 그 검사의 정답 (정오 여부는 저장하지 않고 조회 때 계산한다: 계약 3장)."""
    prediction: Prediction
    true_label: Label

    @property
    def correct(self) -> bool:
        return self.prediction.predicted_label is self.true_label


@dataclass(frozen=True)
class PredictionStats:
    """판정 통계. total·ng_count 는 판정 전체 기준, tn/fp/fn/tp 는 최근 N건 기준 (양성은 NG, 정답은 검사의 true_label)."""
    total: int
    ng_count: int
    tn: int
    fp: int
    fn: int
    tp: int

    @property
    def n(self) -> int:
        return self.tn + self.fp + self.fn + self.tp

    @property
    def ng_rate(self) -> Optional[float]:
        return self.ng_count / self.total if self.total else None

    @property
    def accuracy(self) -> Optional[float]:
        return (self.tn + self.tp) / self.n if self.n else None

    @property
    def recall(self) -> Optional[float]:
        positives = self.tp + self.fn
        return self.tp / positives if positives else None


@dataclass
class ModelVersionRecord:
    """계약 5장 model_version 한 건 (common 에 클래스가 없어 여기서 정의한다)."""
    version: str
    stage: Stage
    metrics: ModelMetrics
    train_data_snapshot: dict
    created_at: datetime
    approved_by: Optional[str]
    approved_at: Optional[datetime]


# ---------- 공통 도우미 ----------

def _page(limit, offset) -> None:
    for name, value, low, high in (("limit", limit, 1, MAX_LIMIT), ("offset", offset, 0, None)):
        if isinstance(value, bool) or not isinstance(value, int) or value < low or (high is not None and value > high):
            bound = f"{low}~{high}" if high is not None else f"{low} 이상"
            raise ValueError(f"{name}: {bound} 정수여야 합니다 (받은 값: {value!r})")


def _enum_value(enum_cls, name: str, value) -> str:
    try:
        return enum_cls(value).value
    except ValueError:
        allowed = ", ".join(m.value for m in enum_cls)
        raise ValueError(f"{name}: 허용값은 {allowed} 입니다 (받은 값: {value!r})") from None


def _unique_violation(error: IntegrityError) -> str:
    """'UNIQUE constraint failed: 테이블.컬럼' 이면 그 뒷부분, 아니면 빈 문자열."""
    text_ = str(error.orig)
    return text_.split("UNIQUE constraint failed:", 1)[1].strip() if "UNIQUE constraint failed" in text_ else ""


# ---------- inspection ----------

def _inspection_to_row(i: Inspection) -> InspectionRow:
    return InspectionRow(
        inspection_id=i.inspection_id, captured_at=i.captured_at, image_path=i.image_path,
        true_label=i.true_label.value, missing_parts=list(i.missing_parts), scenario=i.scenario.value,
        drift_type=i.drift_type, data_split=i.data_split.value)


def _inspection_from_row(r: InspectionRow) -> Inspection:
    return Inspection(
        inspection_id=r.inspection_id, captured_at=r.captured_at, image_path=r.image_path, true_label=r.true_label,
        missing_parts=r.missing_parts, scenario=r.scenario, drift_type=r.drift_type, data_split=r.data_split)


class InspectionRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, inspection: Inspection) -> None:
        """저장한다. 이미 있는 ID 면 DuplicateError (계약 2장: 중복 ID 는 저장하지 않는다)."""
        ensure_aware("captured_at", inspection.captured_at)
        if self.session.get(InspectionRow, inspection.inspection_id) is not None:
            raise DuplicateError(f"이미 있는 inspection_id 입니다: {inspection.inspection_id}")
        self.session.add(_inspection_to_row(inspection))
        try:
            self.session.flush()
        except IntegrityError as e:
            self.session.rollback()
            if _unique_violation(e):
                raise DuplicateError(f"이미 있는 inspection_id 입니다: {inspection.inspection_id}") from e
            raise

    def get(self, inspection_id: str) -> Optional[Inspection]:
        row = self.session.get(InspectionRow, inspection_id)
        return None if row is None else _inspection_from_row(row)

    def exists(self, inspection_id: str) -> bool:
        return self.session.get(InspectionRow, inspection_id) is not None

    @staticmethod
    def _conditions(scenario, data_split, since, until) -> list:
        conditions = []
        if scenario is not None:
            conditions.append(InspectionRow.scenario == _enum_value(Scenario, "scenario", scenario))
        if data_split is not None:
            conditions.append(InspectionRow.data_split == _enum_value(DataSplit, "data_split", data_split))
        if since is not None:  # 시작 포함
            conditions.append(InspectionRow.captured_at >= ensure_aware("since", since))
        if until is not None:  # 끝 미포함
            conditions.append(InspectionRow.captured_at < ensure_aware("until", until))
        return conditions

    def list(self, *, scenario=None, data_split=None, since=None, until=None,
             limit: int = DEFAULT_LIMIT, offset: int = 0) -> list[Inspection]:
        """최신 먼저 (captured_at 내림차순, 같으면 inspection_id 내림차순)."""
        _page(limit, offset)
        stmt = (select(InspectionRow).where(*self._conditions(scenario, data_split, since, until))
                .order_by(InspectionRow.captured_at.desc(), InspectionRow.inspection_id.desc())
                .limit(limit).offset(offset))
        return [_inspection_from_row(r) for r in self.session.scalars(stmt)]

    def count(self, *, scenario=None, data_split=None, since=None, until=None) -> int:
        stmt = select(func.count()).select_from(InspectionRow).where(*self._conditions(scenario, data_split, since, until))
        return self.session.scalar(stmt)

    def max_id(self) -> int:
        """'INS-숫자' 형식 ID 중 가장 큰 번호 (하나도 없으면 0). 다른 형식의 ID 는 무시한다.

        새 검사의 번호를 정할 때 쓴다. 문자열 정렬이 아니라 숫자로 비교하므로 6자리를 넘어가도 맞다.
        """
        stmt = text("SELECT MAX(CAST(SUBSTR(inspection_id, 5) AS INTEGER)) FROM inspection "
                    "WHERE inspection_id GLOB 'INS-[0-9]*' AND SUBSTR(inspection_id, 5) NOT GLOB '*[^0-9]*'")
        return self.session.execute(stmt).scalar() or 0

    def list_for_training(self) -> list[Inspection]:
        """재학습에 쓸 수 있는 검사: data_split 이 train 또는 pool 인 것 전부 (inspection_id 순). 테스트셋은 절대 포함하지 않는다."""
        stmt = (select(InspectionRow).where(InspectionRow.data_split.in_((DataSplit.TRAIN.value, DataSplit.POOL.value)))
                .order_by(InspectionRow.inspection_id))
        return [_inspection_from_row(r) for r in self.session.scalars(stmt)]

    def merge_pool_into_train(self, inspection_ids) -> int:
        """주어진 ID 중 지금 pool 인 검사를 train 으로 편입하고 바꾼 건수를 돌려준다 (계약 2장, C8).

        ID 목록으로만 바꾸므로 그 뒤에 새로 생긴 pool 행은 pool 로 남는다. 이미 train 인 ID 는 건너뛴다.
        """
        ids = list(inspection_ids)
        changed = 0
        for start in range(0, len(ids), 500):  # SQLite 의 변수 개수 한도를 넘지 않게 나눈다
            part = ids[start:start + 500]
            result = self.session.execute(
                update(InspectionRow).where(InspectionRow.inspection_id.in_(part), InspectionRow.data_split == DataSplit.POOL.value)
                .values(data_split=DataSplit.TRAIN.value))
            changed += result.rowcount
        return changed


# ---------- prediction ----------

# prediction_id 는 조회와 삽입을 한 SQL 문으로 처리한다: 현재 최대 번호 + 1 을 같은 문장 안에서 읽고 쓰므로
# 두 요청이 같은 번호를 읽는 일이 없다 (SQLite 는 쓰기 문을 하나씩 처리한다). 최대값은 기본 키 인덱스로 읽는다.
_INSERT_PREDICTION = text("""
INSERT INTO prediction (prediction_id, inspection_id, model_version, predicted_label, score, threshold, latency_ms, predicted_at)
SELECT 'PRD-' || printf('%06d', COALESCE(CAST(SUBSTR((SELECT prediction_id FROM prediction ORDER BY prediction_id DESC LIMIT 1), 5) AS INTEGER), 0) + 1),
       :inspection_id, :model_version, :predicted_label, :score, :threshold, :latency_ms, :predicted_at
RETURNING prediction_id
""")


def _prediction_from_row(r: PredictionRow) -> Prediction:
    return Prediction(
        prediction_id=r.prediction_id, inspection_id=r.inspection_id, model_version=r.model_version,
        predicted_label=r.predicted_label, score=r.score, threshold=r.threshold, latency_ms=r.latency_ms,
        predicted_at=r.predicted_at)


class PredictionRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, output: InferenceOutput, *, inspection_id: str, threshold: float, predicted_at: datetime) -> Prediction:
        """추론 결과 한 건을 prediction_id 를 발급해 저장하고 Prediction 을 돌려준다.

        없는 inspection_id / model_version 이면 NotFoundError, 라벨과 점수가 맞지 않으면 SchemaError (DB 에 쓰지 않는다).
        이 두 경우와 ID 충돌에서는 실패한 문장만 취소되고 세션의 다른 작업은 그대로 남는다.
        일시적 잠금 오류(database is locked)는 최대 3번까지 다시 시도하는데, 그때는 이 세션의 트랜잭션을 롤백한다.
        동시에 쓰는 상황에서는 이 호출을 트랜잭션의 첫 쓰기로 두는 것이 좋다 (요청 하나 = 트랜잭션 하나).
        """
        ensure_aware("predicted_at", predicted_at)
        checked = Prediction(  # 스키마 검증 (번호는 발급 후에 정해지므로 임시 번호)
            prediction_id=f"{PREDICTION_ID_PREFIX}{0:0{PREDICTION_ID_WIDTH}d}", inspection_id=inspection_id,
            model_version=output.model_version, predicted_label=output.predicted_label, score=output.score,
            threshold=threshold, latency_ms=output.latency_ms, predicted_at=predicted_at)
        params = {
            "inspection_id": checked.inspection_id, "model_version": checked.model_version,
            "predicted_label": checked.predicted_label.value, "score": checked.score, "threshold": checked.threshold,
            "latency_ms": checked.latency_ms, "predicted_at": to_utc_text(predicted_at),
        }
        last_error = None
        for _ in range(ID_ATTEMPTS):
            try:
                new_id = self.session.execute(_INSERT_PREDICTION, params).scalar_one()
                break
            except IntegrityError as e:  # SQLite 는 실패한 문장 하나만 취소하므로 세션을 롤백하지 않는다
                if "FOREIGN KEY" in str(e.orig):
                    self._raise_missing_reference(checked)
                    raise
                if not _unique_violation(e):
                    raise
                last_error = e
            except OperationalError as e:
                self.session.rollback()  # 잠금·스냅샷 오류는 새 트랜잭션에서 다시 읽어야 한다
                if "locked" not in str(e.orig).lower() and "busy" not in str(e.orig).lower():
                    raise
                last_error = e
        else:
            raise last_error
        if len(new_id) != len(PREDICTION_ID_PREFIX) + PREDICTION_ID_WIDTH:
            self.session.execute(text("DELETE FROM prediction WHERE prediction_id = :id"), {"id": new_id})  # 방금 넣은 행만 지운다
            raise OverflowError(f"prediction_id 가 {PREDICTION_ID_WIDTH}자리 한도를 넘었습니다")
        return Prediction(
            prediction_id=new_id, inspection_id=checked.inspection_id, model_version=checked.model_version,
            predicted_label=checked.predicted_label, score=checked.score, threshold=checked.threshold,
            latency_ms=checked.latency_ms, predicted_at=predicted_at.astimezone(timezone.utc))  # get() 과 같은 UTC 시각

    def _raise_missing_reference(self, p: Prediction) -> None:
        if self.session.get(InspectionRow, p.inspection_id) is None:
            raise NotFoundError(f"없는 inspection_id 입니다: {p.inspection_id}")
        if self.session.get(ModelVersionRow, p.model_version) is None:
            raise NotFoundError(f"없는 model_version 입니다: {p.model_version}")

    def get(self, prediction_id: str) -> Optional[Prediction]:
        row = self.session.get(PredictionRow, prediction_id)
        return None if row is None else _prediction_from_row(row)

    def _query(self, stmt, model_version, inspection_id, wrong):
        if wrong is not None:
            if not isinstance(wrong, bool):
                raise ValueError(f"wrong: True, False 또는 생략이어야 합니다 (받은 값: {wrong!r})")
            stmt = stmt.join(InspectionRow, PredictionRow.inspection_id == InspectionRow.inspection_id)
            stmt = stmt.where(PredictionRow.predicted_label != InspectionRow.true_label if wrong
                              else PredictionRow.predicted_label == InspectionRow.true_label)
        if model_version is not None:
            stmt = stmt.where(PredictionRow.model_version == model_version)
        if inspection_id is not None:
            stmt = stmt.where(PredictionRow.inspection_id == inspection_id)
        return stmt

    def list(self, *, model_version=None, inspection_id=None, wrong=None,
             limit: int = DEFAULT_LIMIT, offset: int = 0) -> list[Prediction]:
        """최신 먼저 (predicted_at 내림차순, 같으면 prediction_id 내림차순). wrong=True 는 오판만, False 는 정판만."""
        _page(limit, offset)
        stmt = self._query(select(PredictionRow), model_version, inspection_id, wrong)
        stmt = stmt.order_by(PredictionRow.predicted_at.desc(), PredictionRow.prediction_id.desc()).limit(limit).offset(offset)
        return [_prediction_from_row(r) for r in self.session.scalars(stmt)]

    def count(self, *, model_version=None, inspection_id=None, wrong=None) -> int:
        return self.session.scalar(self._query(select(func.count()).select_from(PredictionRow), model_version, inspection_id, wrong))

    @staticmethod
    def _conditions(model_version, inspection_id) -> list:
        conditions = []
        if model_version is not None:
            conditions.append(PredictionRow.model_version == model_version)
        if inspection_id is not None:
            conditions.append(PredictionRow.inspection_id == inspection_id)
        return conditions

    def list_with_truth(self, *, model_version=None, inspection_id=None, wrong=None,
                        limit: int = DEFAULT_LIMIT, offset: int = 0) -> list[PredictionWithTruth]:
        """list() 와 같은 필터·정렬(최신 먼저)이고, 판정마다 검사의 정답(true_label)을 한 번의 JOIN 으로 함께 읽는다."""
        _page(limit, offset)
        if wrong is not None and not isinstance(wrong, bool):
            raise ValueError(f"wrong: True, False 또는 생략이어야 합니다 (받은 값: {wrong!r})")
        stmt = (select(PredictionRow, InspectionRow.true_label)
                .join(InspectionRow, PredictionRow.inspection_id == InspectionRow.inspection_id)
                .where(*self._conditions(model_version, inspection_id)))
        if wrong is not None:
            stmt = stmt.where(PredictionRow.predicted_label != InspectionRow.true_label if wrong
                              else PredictionRow.predicted_label == InspectionRow.true_label)
        stmt = stmt.order_by(PredictionRow.predicted_at.desc(), PredictionRow.prediction_id.desc()).limit(limit).offset(offset)
        return [PredictionWithTruth(_prediction_from_row(row), Label(true_label)) for row, true_label in self.session.execute(stmt)]

    def stats(self, *, recent: int = DEFAULT_RECENT, model_version=None) -> PredictionStats:
        """전체 건수·NG 판정 건수와, 최신 판정 recent 건(list() 와 같은 순서)의 혼동행렬 네 칸을 센다.

        최근 구간은 SQL 의 ORDER BY ... LIMIT 서브쿼리로 자르고 그 안에서 CASE 로 센다 (행을 파이썬으로 가져오지 않는다).
        """
        if isinstance(recent, bool) or not isinstance(recent, int) or not 1 <= recent <= MAX_LIMIT:
            raise ValueError(f"recent: 1~{MAX_LIMIT} 정수여야 합니다 (받은 값: {recent!r})")
        conditions = self._conditions(model_version, None)
        total, ng_count = self.session.execute(
            select(func.count(), func.coalesce(func.sum(case((PredictionRow.predicted_label == "NG", 1), else_=0)), 0))
            .select_from(PredictionRow).where(*conditions)).one()
        window = (select(PredictionRow.predicted_label.label("p"), InspectionRow.true_label.label("t"))
                  .select_from(PredictionRow).join(InspectionRow, PredictionRow.inspection_id == InspectionRow.inspection_id)
                  .where(*conditions)
                  .order_by(PredictionRow.predicted_at.desc(), PredictionRow.prediction_id.desc()).limit(recent).subquery())

        def cell(predicted, truth):
            return func.coalesce(func.sum(case((and_(window.c.p == predicted, window.c.t == truth), 1), else_=0)), 0)

        tn, fp, fn, tp = self.session.execute(select(cell("OK", "OK"), cell("NG", "OK"), cell("OK", "NG"), cell("NG", "NG"))).one()
        return PredictionStats(total=total, ng_count=ng_count, tn=tn, fp=fp, fn=fn, tp=tp)

    def list_for_window(self, *, since: datetime, until: datetime, model_version=None) -> list[tuple]:
        """[since, until) 에 판정한 건의 (predicted_at, model_version, predicted_label, true_label) 를 시간순으로 돌려준다 (모니터링 구간 묶음용).

        data_split 으로 거르지 않는다. 판정마다 한 줄이다 (같은 검사를 여러 번 판정하면 여러 줄).
        """
        ensure_aware("since", since)
        ensure_aware("until", until)
        stmt = (select(PredictionRow.predicted_at, PredictionRow.model_version, PredictionRow.predicted_label, InspectionRow.true_label)
                .join(InspectionRow, PredictionRow.inspection_id == InspectionRow.inspection_id)
                .where(PredictionRow.predicted_at >= since, PredictionRow.predicted_at < until, *self._conditions(model_version, None))
                .order_by(PredictionRow.predicted_at.asc(), PredictionRow.prediction_id.asc()))
        return [tuple(row) for row in self.session.execute(stmt)]

    def recent_image_paths(self, *, recent: int, model_version=None) -> list[tuple[str, str]]:
        """최신 판정 recent 건(stats 와 같은 순서)의 (inspection_id, image_path). 같은 검사를 여러 번 판정했으면 그만큼 나온다."""
        if isinstance(recent, bool) or not isinstance(recent, int) or not 1 <= recent <= MAX_LIMIT:
            raise ValueError(f"recent: 1~{MAX_LIMIT} 정수여야 합니다 (받은 값: {recent!r})")
        stmt = (select(PredictionRow.inspection_id, InspectionRow.image_path)
                .join(InspectionRow, PredictionRow.inspection_id == InspectionRow.inspection_id)
                .where(*self._conditions(model_version, None))
                .order_by(PredictionRow.predicted_at.desc(), PredictionRow.prediction_id.desc()).limit(recent))
        return [(inspection_id, image_path) for inspection_id, image_path in self.session.execute(stmt)]


# ---------- model_version ----------

def _record_from_row(r: ModelVersionRow) -> ModelVersionRecord:
    return ModelVersionRecord(
        version=r.version, stage=Stage(r.stage), metrics=ModelMetrics.from_dict(r.metrics),
        train_data_snapshot=dict(r.train_data_snapshot), created_at=r.created_at,
        approved_by=r.approved_by, approved_at=r.approved_at)


class ModelVersionRepository:
    def __init__(self, session: Session):
        self.session = session

    def register(self, metrics: ModelMetrics, *, stage: Stage = Stage.CANDIDATE,
                 approved_by: Optional[str] = None, approved_at: Optional[datetime] = None) -> ModelVersionRecord:
        """모델 버전을 등록한다. metrics 하나에서 metrics, train_data_snapshot, created_at 을 모두 채워 서로 일치한다.

        이미 있는 버전이면 DuplicateError, production 이 이미 있는데 production 으로 등록하면 ConflictError.
        """
        if not isinstance(metrics, ModelMetrics):
            raise TypeError(f"metrics: ModelMetrics 여야 합니다 (받은 값: {type(metrics).__name__})")
        stage = Stage(stage)
        ensure_aware("created_at", metrics.created_at)
        if approved_at is not None:
            ensure_aware("approved_at", approved_at)
        if approved_by is not None and not isinstance(approved_by, str):
            raise ValueError(f"approved_by: 문자열이어야 합니다 (받은 값: {approved_by!r})")

        if self.session.get(ModelVersionRow, metrics.version) is not None:
            raise DuplicateError(f"이미 있는 버전입니다: {metrics.version}")
        if stage is Stage.PRODUCTION:
            current = self.session.scalar(select(ModelVersionRow.version).where(ModelVersionRow.stage == Stage.PRODUCTION.value))
            if current is not None:
                raise ConflictError(f"production 이 이미 있습니다: {current}")

        row = ModelVersionRow(
            version=metrics.version, stage=stage.value, metrics=metrics.to_dict(),
            train_data_snapshot=dict(metrics.train_data_snapshot), created_at=metrics.created_at,
            approved_by=approved_by, approved_at=approved_at)
        self.session.add(row)
        try:
            self.session.flush()
        except IntegrityError as e:
            self.session.rollback()
            violated = _unique_violation(e)
            if violated.endswith("model_version.version"):
                raise DuplicateError(f"이미 있는 버전입니다: {metrics.version}") from e
            if violated.endswith("model_version.stage"):
                raise ConflictError("production 이 이미 있습니다") from e
            raise
        return _record_from_row(row)

    def get(self, version: str) -> Optional[ModelVersionRecord]:
        row = self.session.get(ModelVersionRow, version)
        return None if row is None else _record_from_row(row)

    def list(self) -> list[ModelVersionRecord]:
        """오래된 버전부터 (created_at 오름차순, 같으면 version 오름차순)."""
        stmt = select(ModelVersionRow).order_by(ModelVersionRow.created_at.asc(), ModelVersionRow.version.asc())
        return [_record_from_row(r) for r in self.session.scalars(stmt)]

    def get_production(self) -> Optional[ModelVersionRecord]:
        row = self.session.scalar(select(ModelVersionRow).where(ModelVersionRow.stage == Stage.PRODUCTION.value))
        return None if row is None else _record_from_row(row)

    def _row(self, version: str) -> ModelVersionRow:
        row = self.session.get(ModelVersionRow, version)
        if row is None:
            raise NotFoundError(f"없는 버전입니다: {version}")
        return row

    def transition(self, version: str, to: Stage, *, approved_by: Optional[str] = None,
                   approved_at: Optional[datetime] = None) -> ModelVersionRecord:
        """계약 5장의 전이표(ALLOWED_TRANSITIONS)대로 stage 를 바꾼다. 표에 없는 전이는 InvalidTransitionError.

        production 으로 올릴 때는 approved_by·approved_at 이 필수이고 (마지막으로 올린 사람과 시각, C7),
        지금 production 이 따로 있으면 ConflictError 다 (먼저 archive_production 으로 내려야 한다).
        """
        to = Stage(to)
        row = self._row(version)
        current = Stage(row.stage)
        if (current, to) not in ALLOWED_TRANSITIONS:
            raise InvalidTransitionError(f"{version}: {current.value} 에서 {to.value} 로는 바꿀 수 없습니다")
        if to is Stage.PRODUCTION:
            if not isinstance(approved_by, str) or not approved_by.strip():
                raise ValueError("approved_by: production 으로 올릴 때는 필수입니다")
            ensure_aware("approved_at", approved_at)
            other = self.get_production()
            if other is not None:
                raise ConflictError(f"production 이 이미 있습니다: {other.version}")
            row.approved_by, row.approved_at = approved_by, approved_at
        row.stage = to.value
        self._flush_stage(row)
        return _record_from_row(row)

    def archive_production(self) -> Optional[ModelVersionRecord]:
        """지금 production 을 archived 로 내린다 (승격·롤백이 새 production 을 올리기 전에 먼저 부른다). 없으면 None."""
        row = self.session.scalar(select(ModelVersionRow).where(ModelVersionRow.stage == Stage.PRODUCTION.value))
        if row is None:
            return None
        row.stage = Stage.ARCHIVED.value
        self._flush_stage(row)
        return _record_from_row(row)

    def _flush_stage(self, row: ModelVersionRow) -> None:
        try:
            self.session.flush()
        except IntegrityError as e:
            if _unique_violation(e).endswith("model_version.stage"):
                raise ConflictError("production 이 이미 있습니다") from e
            raise


# ---------- 파트 3 보조 테이블: approval_request, training_run, deployment, audit_log (계약 6~7장, C7) ----------

ID_WIDTH = 6  # APR-000001 처럼 접두어 + 6자리 (PRD- 와 같다)


def _insert_with_next_id(session: Session, table: str, id_column: str, prefix: str, values: dict, on_foreign_key) -> str:
    """ID 를 발급해 한 행을 넣고 ID 를 돌려준다. PredictionRepository.add 와 같은 방식이다.

    현재 최대 번호 + 1 을 읽는 일과 삽입을 SQL 한 문장으로 처리해 두 요청이 같은 번호를 읽지 않게 한다.
    테이블·컬럼 이름은 이 파일의 고정 값만 받는다 (사용자 입력이 들어가지 않는다). 값은 모두 바인드 인자다.
    FK 위반이면 on_foreign_key() 가 NotFoundError 를 던진다. ID 충돌과 잠금 오류는 최대 3번 다시 시도한다.
    """
    columns = list(values)
    quoted = ", ".join(f'"{c}"' for c in columns)  # trigger 처럼 SQL 예약어인 이름이 있어 따옴표로 감싼다
    binds = ", ".join(f":{c}" for c in columns)
    next_number = (f"COALESCE(CAST(SUBSTR((SELECT {id_column} FROM {table} ORDER BY {id_column} DESC LIMIT 1), "
                   f"{len(prefix) + 1}) AS INTEGER), 0) + 1")
    statement = text(f"INSERT INTO {table} ({id_column}, {quoted}) "
                     f"SELECT '{prefix}' || printf('%0{ID_WIDTH}d', {next_number}), {binds} RETURNING {id_column}")
    last_error = None
    for _ in range(ID_ATTEMPTS):
        try:
            new_id = session.execute(statement, values).scalar_one()
            break
        except IntegrityError as e:
            if "FOREIGN KEY" in str(e.orig):
                on_foreign_key()
                raise
            if not _unique_violation(e).endswith(f"{table}.{id_column}"):
                raise  # 번호 충돌이 아닌 제약 위반은 그대로 올린다
            last_error = e
        except OperationalError as e:
            session.rollback()  # 잠금·스냅샷 오류는 새 트랜잭션에서 다시 읽어야 한다
            if "locked" not in str(e.orig).lower() and "busy" not in str(e.orig).lower():
                raise
            last_error = e
    else:
        raise last_error
    if len(new_id) != len(prefix) + ID_WIDTH:
        session.execute(text(f"DELETE FROM {table} WHERE {id_column} = :id"), {"id": new_id})  # 방금 넣은 행만 지운다
        raise OverflowError(f"{table} 의 ID 가 {ID_WIDTH}자리 한도를 넘었습니다")
    return new_id


def _utc_text(name: str, value) -> str:
    return to_utc_text(ensure_aware(name, value))


def _text(name: str, value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}: 비어 있지 않은 문자열이어야 합니다 (받은 값: {value!r})")
    return value


def _reason_required(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("reason: 사유를 입력해야 합니다")
    return value


def _version_check(session: Session, version: str):
    """없는 model_version 이면 NotFoundError 를 던지는 함수 (FK 위반 때 쓴다)."""
    def raise_if_missing():
        if session.get(ModelVersionRow, version) is None:
            raise NotFoundError(f"없는 model_version 입니다: {version}")
    return raise_if_missing


# ---------- approval_request ----------

@dataclass
class ApprovalRecord:
    approval_id: str
    model_version: str
    requested_by: str
    status: ApprovalStatus
    reason: Optional[str]
    handled_by: Optional[str]
    handled_at: Optional[datetime]
    created_at: datetime


def _approval_from_row(r: ApprovalRequestRow) -> ApprovalRecord:
    return ApprovalRecord(
        approval_id=r.approval_id, model_version=r.model_version, requested_by=r.requested_by,
        status=ApprovalStatus(r.status), reason=r.reason, handled_by=r.handled_by, handled_at=r.handled_at,
        created_at=r.created_at)


class ApprovalRepository:
    def __init__(self, session: Session):
        self.session = session

    def add_pending(self, model_version: str, *, requested_by: str, created_at: datetime) -> ApprovalRecord:
        """pending 승인 요청을 만든다. 같은 버전에 pending 이 이미 있으면 ConflictError (계약 6장), 없는 버전이면 NotFoundError."""
        values = {"model_version": model_version, "requested_by": _text("requested_by", requested_by),
                  "status": ApprovalStatus.PENDING.value, "reason": None, "handled_by": None, "handled_at": None,
                  "created_at": _utc_text("created_at", created_at)}
        check_version = _version_check(self.session, model_version)
        check_version()
        pending = self.session.scalar(select(ApprovalRequestRow.approval_id).where(
            ApprovalRequestRow.model_version == model_version, ApprovalRequestRow.status == ApprovalStatus.PENDING.value))
        if pending is not None:
            raise ConflictError(f"{model_version} 에는 처리 대기 중인 승인 요청이 이미 있습니다: {pending}")
        try:
            new_id = _insert_with_next_id(self.session, "approval_request", "approval_id", "APR-", values, check_version)
        except IntegrityError as e:
            if _unique_violation(e).endswith("approval_request.model_version"):
                raise ConflictError(f"{model_version} 에는 처리 대기 중인 승인 요청이 이미 있습니다") from e
            raise
        return _approval_from_row(self.session.get(ApprovalRequestRow, new_id))

    def get(self, approval_id: str) -> Optional[ApprovalRecord]:
        row = self.session.get(ApprovalRequestRow, approval_id)
        return None if row is None else _approval_from_row(row)

    def list(self, *, status=None, limit: int = DEFAULT_LIMIT, offset: int = 0) -> list[ApprovalRecord]:
        """최신 먼저 (created_at 내림차순, 같으면 approval_id 내림차순)."""
        _page(limit, offset)
        stmt = select(ApprovalRequestRow)
        if status is not None:
            stmt = stmt.where(ApprovalRequestRow.status == _enum_value(ApprovalStatus, "status", status))
        stmt = stmt.order_by(ApprovalRequestRow.created_at.desc(), ApprovalRequestRow.approval_id.desc()).limit(limit).offset(offset)
        return [_approval_from_row(r) for r in self.session.scalars(stmt)]

    def count(self, *, status=None) -> int:
        stmt = select(func.count()).select_from(ApprovalRequestRow)
        if status is not None:
            stmt = stmt.where(ApprovalRequestRow.status == _enum_value(ApprovalStatus, "status", status))
        return self.session.scalar(stmt)

    def mark_handled(self, approval_id: str, *, status: ApprovalStatus, handled_by: str, handled_at: datetime,
                     reason: Optional[str] = None) -> ApprovalRecord:
        """pending 요청을 approved 또는 rejected 로 처리한다. 이미 처리된 요청이면 InvalidTransitionError, 없으면 NotFoundError.

        거절(rejected)은 사유가 필수다 (계약 6장).
        """
        status = ApprovalStatus(status)
        if status is ApprovalStatus.PENDING:
            raise ValueError("status: approved 또는 rejected 여야 합니다")
        if status is ApprovalStatus.REJECTED:
            _reason_required(reason)
        _text("handled_by", handled_by)
        ensure_aware("handled_at", handled_at)
        row = self.session.get(ApprovalRequestRow, approval_id)
        if row is None:
            raise NotFoundError(f"없는 approval_id 입니다: {approval_id}")
        if row.status != ApprovalStatus.PENDING.value:
            raise InvalidTransitionError(f"{approval_id}: 이미 처리된 승인 요청입니다 ({row.status})")
        row.status, row.reason, row.handled_by, row.handled_at = status.value, reason, handled_by, handled_at
        self.session.flush()
        return _approval_from_row(row)


# ---------- training_run ----------

@dataclass
class TrainingRunRecord:
    run_id: str
    status: RunStatus
    trigger: RunTrigger
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    result_version: Optional[str]
    step: Optional[RunStep]
    error_message: Optional[str]
    gate_result: Optional[dict]


def _run_from_row(r: TrainingRunRow) -> TrainingRunRecord:
    return TrainingRunRecord(
        run_id=r.run_id, status=RunStatus(r.status), trigger=RunTrigger(r.trigger), started_at=r.started_at,
        finished_at=r.finished_at, result_version=r.result_version, step=None if r.step is None else RunStep(r.step),
        error_message=r.error_message, gate_result=None if r.gate_result is None else dict(r.gate_result))


class TrainingRunRepository:
    UPDATABLE = ("status", "step", "started_at", "finished_at", "result_version", "error_message", "gate_result")

    def __init__(self, session: Session):
        self.session = session

    def add(self, trigger: RunTrigger) -> TrainingRunRecord:
        """대기(queued) 상태의 실행 기록을 만든다."""
        values = {"status": RunStatus.QUEUED.value, "trigger": _enum_value(RunTrigger, "trigger", trigger), "started_at": None,
                  "finished_at": None, "result_version": None, "step": None, "error_message": None, "gate_result": None}
        new_id = _insert_with_next_id(self.session, "training_run", "run_id", "RUN-", values, lambda: None)
        return _run_from_row(self.session.get(TrainingRunRow, new_id))

    def get(self, run_id: str) -> Optional[TrainingRunRecord]:
        row = self.session.get(TrainingRunRow, run_id)
        return None if row is None else _run_from_row(row)

    def list(self, *, status=None, limit: int = DEFAULT_LIMIT, offset: int = 0) -> list[TrainingRunRecord]:
        """최신 먼저 (run_id 내림차순)."""
        _page(limit, offset)
        stmt = select(TrainingRunRow)
        if status is not None:
            stmt = stmt.where(TrainingRunRow.status == _enum_value(RunStatus, "status", status))
        stmt = stmt.order_by(TrainingRunRow.run_id.desc()).limit(limit).offset(offset)
        return [_run_from_row(r) for r in self.session.scalars(stmt)]

    def count(self, *, status=None) -> int:
        stmt = select(func.count()).select_from(TrainingRunRow)
        if status is not None:
            stmt = stmt.where(TrainingRunRow.status == _enum_value(RunStatus, "status", status))
        return self.session.scalar(stmt)

    def get_by_result_version(self, version: str) -> Optional[TrainingRunRecord]:
        """result_version 이 이 버전인 기록 (없으면 None). 승인 때 후보의 게이트 기준 모델을 확인하는 데 쓴다."""
        row = self.session.scalar(select(TrainingRunRow).where(TrainingRunRow.result_version == version)
                                  .order_by(TrainingRunRow.run_id.desc()).limit(1))
        return None if row is None else _run_from_row(row)

    def active(self) -> Optional[TrainingRunRecord]:
        """대기(queued) 또는 실행(running) 중인 기록 하나 (없으면 None). 동시 학습 방지에 쓴다."""
        row = self.session.scalar(select(TrainingRunRow).where(
            TrainingRunRow.status.in_((RunStatus.QUEUED.value, RunStatus.RUNNING.value))).order_by(TrainingRunRow.run_id).limit(1))
        return None if row is None else _run_from_row(row)

    def fail_unfinished(self, message: str, finished_at: datetime) -> int:
        """queued·running 으로 남은 기록을 모두 failed 로 바꾸고 건수를 돌려준다 (서버 재시작 정리, api-v1.md 4장)."""
        _text("message", message)
        stamp = ensure_aware("finished_at", finished_at).astimezone(timezone.utc)
        result = self.session.execute(
            update(TrainingRunRow).where(TrainingRunRow.status.in_((RunStatus.QUEUED.value, RunStatus.RUNNING.value)))
            .values(status=RunStatus.FAILED.value, error_message=message, finished_at=stamp))
        return result.rowcount

    def update(self, run_id: str, **fields) -> TrainingRunRecord:
        """넘긴 항목만 바꾼다 (UPDATABLE 항목). 없는 run_id 나 없는 result_version 은 NotFoundError."""
        unknown = set(fields) - set(self.UPDATABLE)
        if unknown:
            raise ValueError(f"바꿀 수 없는 항목입니다: {', '.join(sorted(unknown))}")
        row = self.session.get(TrainingRunRow, run_id)
        if row is None:
            raise NotFoundError(f"없는 run_id 입니다: {run_id}")
        if "status" in fields:
            fields["status"] = _enum_value(RunStatus, "status", fields["status"])
        if fields.get("step") is not None:
            fields["step"] = _enum_value(RunStep, "step", fields["step"])
        for name in ("started_at", "finished_at"):
            if fields.get(name) is not None:
                fields[name] = ensure_aware(name, fields[name]).astimezone(timezone.utc)  # 저장 후 읽을 때와 같은 UTC 시각
        if fields.get("result_version") is not None:
            _version_check(self.session, fields["result_version"])()
        for name, value in fields.items():
            setattr(row, name, value)
        self.session.flush()
        return _run_from_row(row)


# ---------- deployment ----------

@dataclass
class DeploymentRecord:
    deployment_id: str
    version: str
    action: DeploymentAction
    reason: Optional[str]
    actor: str
    at: datetime


def _deployment_from_row(r: DeploymentRow) -> DeploymentRecord:
    return DeploymentRecord(deployment_id=r.deployment_id, version=r.version, action=DeploymentAction(r.action),
                            reason=r.reason, actor=r.actor, at=r.at)


class DeploymentRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, version: str, action: DeploymentAction, *, actor: str, at: datetime,
            reason: Optional[str] = None) -> DeploymentRecord:
        """배포·롤백 이력 한 건. 롤백은 사유가 필수다. 없는 버전이면 NotFoundError."""
        action = DeploymentAction(action)
        if action is DeploymentAction.ROLLBACK:
            _reason_required(reason)
        values = {"version": version, "action": action.value, "reason": reason, "actor": _text("actor", actor),
                  "at": _utc_text("at", at)}
        new_id = _insert_with_next_id(self.session, "deployment", "deployment_id", "DEP-", values, _version_check(self.session, version))
        return _deployment_from_row(self.session.get(DeploymentRow, new_id))

    @staticmethod
    def _conditions(version, action) -> list:
        conditions = []
        if version is not None:
            conditions.append(DeploymentRow.version == version)
        if action is not None:
            conditions.append(DeploymentRow.action == _enum_value(DeploymentAction, "action", action))
        return conditions

    def list(self, *, version=None, action=None, limit: int = DEFAULT_LIMIT, offset: int = 0) -> list[DeploymentRecord]:
        """최신 먼저 (at 내림차순, 같으면 deployment_id 내림차순)."""
        _page(limit, offset)
        stmt = (select(DeploymentRow).where(*self._conditions(version, action))
                .order_by(DeploymentRow.at.desc(), DeploymentRow.deployment_id.desc()).limit(limit).offset(offset))
        return [_deployment_from_row(r) for r in self.session.scalars(stmt)]

    def count(self, *, version=None, action=None) -> int:
        return self.session.scalar(select(func.count()).select_from(DeploymentRow).where(*self._conditions(version, action)))


# ---------- audit_log ----------

@dataclass
class AuditLogRecord:
    log_id: str
    actor: str
    role: Role
    action: AuditAction
    target: str
    at: datetime


def _audit_from_row(r: AuditLogRow) -> AuditLogRecord:
    return AuditLogRecord(log_id=r.log_id, actor=r.actor, role=Role(r.role), action=AuditAction(r.action),
                          target=r.target, at=r.at)


class AuditLogRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, *, actor: str, role: Role, action: AuditAction, target: str, at: datetime) -> AuditLogRecord:
        """감사 기록 한 건 (role 과 action 은 계약 7장 C7 의 값만)."""
        values = {"actor": _text("actor", actor), "role": _enum_value(Role, "role", role),
                  "action": _enum_value(AuditAction, "action", action), "target": _text("target", target),
                  "at": _utc_text("at", at)}
        new_id = _insert_with_next_id(self.session, "audit_log", "log_id", "LOG-", values, lambda: None)
        return _audit_from_row(self.session.get(AuditLogRow, new_id))

    def list(self, *, action=None, limit: int = DEFAULT_LIMIT, offset: int = 0) -> list[AuditLogRecord]:
        """최신 먼저 (at 내림차순, 같으면 log_id 내림차순)."""
        _page(limit, offset)
        stmt = select(AuditLogRow)
        if action is not None:
            stmt = stmt.where(AuditLogRow.action == _enum_value(AuditAction, "action", action))
        stmt = stmt.order_by(AuditLogRow.at.desc(), AuditLogRow.log_id.desc()).limit(limit).offset(offset)
        return [_audit_from_row(r) for r in self.session.scalars(stmt)]

    def count(self, *, action=None) -> int:
        stmt = select(func.count()).select_from(AuditLogRow)
        if action is not None:
            stmt = stmt.where(AuditLogRow.action == _enum_value(AuditAction, "action", action))
        return self.session.scalar(stmt)
