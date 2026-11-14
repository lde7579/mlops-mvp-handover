"""백엔드 테스트 도우미: 값으로 도메인 객체를 만들고(이미지·Collector·torch 를 쓰지 않는다), 원시 SQL 로 행을 넣는다."""
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from backend.repository import InspectionRepository, ModelVersionRepository, Stage
from common.schemas import InferenceOutput, Inspection, ModelMetrics

KST = timezone(timedelta(hours=9))
T0 = datetime(2026, 10, 18, 9, 0, 0, tzinfo=KST)


def at(minutes: int = 0) -> datetime:
    return T0 + timedelta(minutes=minutes)


def make_inspection(n: int = 1, **over) -> Inspection:
    values = {
        "inspection_id": f"INS-{n:06d}", "captured_at": at(n), "image_path": f"data/pool/INS-{n:06d}.png",
        "true_label": "NG", "missing_parts": [2], "scenario": "normal", "data_split": "pool",
    }
    values.update(over)
    return Inspection(**values)


def make_metrics(version: str = "v1", created_minutes: int = 0, **over) -> ModelMetrics:
    values = {
        "version": version, "threshold": 0.5,
        "normal": {"accuracy": 1.0, "recall": 1.0, "precision": 1.0, "f1": 1.0,
                   "confusion_matrix": [[400, 0], [0, 100]], "n": 500},
        "drift": {"accuracy": 0.9, "recall": 1.0, "precision": 0.8, "f1": 0.9,
                  "confusion_matrix": [[350, 50], [0, 100]], "n": 500},
        "train_data_snapshot": {"train": 2000, "pool": 0, "seed": 42},
        "created_at": at(created_minutes).isoformat(),
    }
    values.update(over)
    return ModelMetrics.from_dict(values)


def make_output(label: str = "NG", score: float = 0.9, version: str = "v1", latency_ms: int = 3) -> InferenceOutput:
    return InferenceOutput(predicted_label=label, score=score, model_version=version, latency_ms=latency_ms)


def seed(session, inspections: int = 3, versions=("v1",)) -> None:
    """v1 을 production 으로(나머지는 candidate), inspection 을 n 건 넣는다. inspection 1·2 는 NG, 3 이후는 OK."""
    models = ModelVersionRepository(session)
    for i, v in enumerate(versions):
        models.register(make_metrics(v, created_minutes=i), stage=Stage.PRODUCTION if i == 0 else Stage.CANDIDATE)
    repo = InspectionRepository(session)
    for n in range(1, inspections + 1):
        ng = n <= 2
        repo.add(make_inspection(n, true_label="NG" if ng else "OK", missing_parts=[2] if ng else []))
    session.flush()


# ---------- 원시 SQL (스키마·저장소를 거치지 않고 DB 제약을 확인할 때) ----------

def raw_inspection(conn, **over) -> None:
    values = {"id": "INS-000001", "t": "2026-10-18T00:00:00.000000+00:00", "path": "data/pool/INS-000001.png",
              "label": "NG", "parts": "[2]", "scenario": "normal", "drift": None, "split": "pool"}
    values.update(over)
    conn.execute(text(
        "INSERT INTO inspection (inspection_id, captured_at, image_path, true_label, missing_parts, scenario, drift_type, data_split) "
        "VALUES (:id, :t, :path, :label, :parts, :scenario, :drift, :split)"), values)


def raw_model_version(conn, **over) -> None:
    values = {"version": "v1", "stage": "candidate", "metrics": "{}", "snap": "{}", "t": "2026-10-18T00:00:00.000000+00:00",
              "by": None, "at": None}
    values.update(over)
    conn.execute(text(
        "INSERT INTO model_version (version, stage, metrics, train_data_snapshot, created_at, approved_by, approved_at) "
        "VALUES (:version, :stage, :metrics, :snap, :t, :by, :at)"), values)


def raw_prediction(conn, **over) -> None:
    values = {"id": "PRD-000001", "ins": "INS-000001", "ver": "v1", "label": "NG", "score": 0.9, "threshold": 0.5,
              "latency": 3, "t": "2026-10-18T00:00:00.000000+00:00"}
    values.update(over)
    conn.execute(text(
        "INSERT INTO prediction (prediction_id, inspection_id, model_version, predicted_label, score, threshold, latency_ms, predicted_at) "
        "VALUES (:id, :ins, :ver, :label, :score, :threshold, :latency, :t)"), values)


# ---------- 3-B: 앱·설정·인증 도우미 ----------
import asyncio
from typing import Annotated

from fastapi import Depends, FastAPI

from backend.deps import CurrentUser, User, require_role
from backend.errors import ApiError
from backend.security import Role, create_token, hash_password
from backend.settings import Account, Settings

SECRET = "0123456789abcdef" * 4          # 테스트용 고정 비밀키 (64자)
OTHER_SECRET = "fedcba9876543210" * 4
FAST_ITERATIONS = 1000                    # 테스트 계정 해시는 반복을 줄여 빠르게 (실제 설정은 600,000회)
PASSWORDS = {"operator": "operator-pw", "admin": "admin-pw"}


def make_settings(**over) -> Settings:
    accounts = (
        Account("operator", Role.OPERATOR, hash_password(PASSWORDS["operator"], iterations=FAST_ITERATIONS)),
        Account("admin", Role.ADMIN, hash_password(PASSWORDS["admin"], iterations=FAST_ITERATIONS)),
    )
    values = {"db_path": ":memory:", "token_minutes": 480, "accounts": accounts, "secret_key": SECRET}
    values.update(over)
    return Settings(**values)


def bearer(settings: Settings, username: str, now=None, secret=None) -> dict:
    token = create_token(username, secret or settings.secret_key, settings.token_minutes, now)
    return {"Authorization": f"Bearer {token}"}


def add_protected_routes(app: FastAPI) -> None:
    """권한 확인 의존성을 시험하는 임시 경로 (실제 엔드포인트가 아니다)."""

    @app.get("/_test/any")
    def any_user(user: CurrentUser):
        return {"username": user.username, "role": user.role}

    @app.get("/_test/operator")
    def operator_or_admin(user: Annotated[User, Depends(require_role(Role.OPERATOR))]):
        return {"username": user.username, "role": user.role}

    @app.get("/_test/admin")
    def admin_only(user: Annotated[User, Depends(require_role(Role.ADMIN))]):
        return {"username": user.username, "role": user.role}


def add_error_routes(app: FastAPI) -> None:
    @app.get("/_test/boom")
    def boom():
        raise RuntimeError("boom-내부-비밀")

    @app.get("/_test/conflict")
    def conflict():
        raise ApiError(409)

    @app.get("/_test/forbidden")
    def forbidden():
        raise ApiError(403, "직접 쓴 메시지")

    @app.get("/_test/number")
    def number(x: int):
        return {"x": x}


def asgi_call(app: FastAPI, method: str, path: str, on_response_start, headers: dict = None, body: bytes = b"") -> None:
    """TestClient 없이 앱을 ASGI 로 직접 호출한다. 응답이 나가기 시작하는 순간 on_response_start(status) 를 부른다
    (TestClient 는 앱이 완전히 끝날 때까지 기다려서, 응답 전/후의 차이를 볼 수 없다). headers 는 3-C, body 는 3-D 에서 추가 (기본은 비어 있음)."""
    extra = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]

    async def run():
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "scheme": "http",
                 "path": path, "raw_path": path.encode(), "query_string": b"", "headers": [(b"host", b"test"), *extra],
                 "client": ("127.0.0.1", 1), "server": ("test", 80)}

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            if message["type"] == "http.response.start":
                on_response_start(message["status"])

        await app(scope, receive, send)

    asyncio.run(run())


# ---------- 3-C: 이미지 파일·폴더 링크 도우미 ----------
import io
import os
import subprocess
from pathlib import Path

from PIL import Image


def png_bytes(value: int = 200, size: int = 8) -> bytes:
    """작은 흑백 PNG (수집기를 쓰지 않는다)."""
    buf = io.BytesIO()
    Image.new("L", (size, size), value).save(buf, format="PNG")
    return buf.getvalue()


def write_file(root, rel: str, data: bytes = b"x") -> Path:
    path = Path(root) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def make_dir_link(link, target) -> None:
    """폴더 연결을 만든다: Windows 는 정션(mklink /J, 관리자 권한 불필요), 그 밖은 심볼릭 링크.
    만들지 못하면 skip 하지 않고 실패시킨다 (환경이 보안 검사를 못 하게 되는 것을 숨기지 않기 위해)."""
    link, target = Path(link), Path(target)
    if os.name == "nt":
        done = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True)
        assert done.returncode == 0, f"정션을 만들지 못했습니다: {done.stdout} {done.stderr}"
    else:
        os.symlink(target, link, target_is_directory=True)
    assert link.exists()


# ---------- 3-D: 모델 폴더 도우미 ----------
import json
import shutil


def copy_models(src_root, dest_root, versions=("v1",)) -> None:
    """학습해 둔 모델(세션 fixture)을 시험용 폴더의 models/ 로 복사한다."""
    for v in versions:
        shutil.copytree(Path(src_root) / "models" / v, Path(dest_root) / "models" / v)


def clone_version(root, src: str = "v1", dst: str = "v2", **metrics_over) -> None:
    """models/<src> 를 models/<dst> 로 복사하고 metrics.json 의 version(과 바꿀 항목)만 고친다. 가중치는 같다."""
    source, target = Path(root) / "models" / src, Path(root) / "models" / dst
    shutil.copytree(source, target)
    metrics = json.loads((target / "metrics.json").read_text(encoding="utf-8"))
    metrics.update(metrics_over, version=dst)
    (target / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")


def write_png(root, rel: str, array) -> Path:
    path = Path(root) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="PNG")
    return path


# ---------- 3-D-2: 읽기 전용 조회 시험용 시드 (모델도 이미지도 만들지 않고 행만 직접 넣는다) ----------

def utc_text(minutes: int) -> str:
    """at(minutes) 시각을 저장소와 같은 고정 폭 UTC 문자열로."""
    from backend.database import to_utc_text

    return to_utc_text(at(minutes))


def insert_predictions(conn, rows) -> None:
    """판정 행을 한 번에 넣는다. rows: {"id", "ins", "ver", "label", "minutes"} 의 목록. 점수는 라벨에서 정한다 (NG 0.9, OK 0.1, 임계값 0.5)."""
    params = [{"id": r["id"], "ins": r["ins"], "ver": r["ver"], "label": r["label"], "score": 0.9 if r["label"] == "NG" else 0.1,
               "threshold": 0.5, "latency": 3, "t": utc_text(r["minutes"])} for r in rows]
    conn.execute(text(
        "INSERT INTO prediction (prediction_id, inspection_id, model_version, predicted_label, score, threshold, latency_ms, predicted_at) "
        "VALUES (:id, :ins, :ver, :label, :score, :threshold, :latency, :t)"), params)


# 검사 1~6 은 정답 NG, 7~12 는 정답 OK. 판정 16건 (표시는 정답 -> 판정 = 혼동행렬 칸):
#  v1 (at(101..110), PRD-1..10): 1 NG->NG TP, 2 NG->NG TP, 3 NG->OK FN, 7 OK->OK TN, 8 OK->NG FP,
#                                4 NG->NG TP, 9 OK->OK TN, 5 NG->OK FN, 10 OK->OK TN, 6 NG->NG TP
#  v2 (at(201..204), PRD-11..14): 11 OK->OK TN, 12 OK->NG FP, 11 OK->OK TN, 7 OK->OK TN
#  v3 (모두 at(300), PRD-15·16 같은 시각): 1 NG->NG TP, 7 OK->NG FP
PREDICTION_SEED = (
    [dict(id=f"PRD-{k:06d}", ins=f"INS-{i:06d}", ver="v1", label=label, minutes=100 + k)
     for k, (i, label) in enumerate([(1, "NG"), (2, "NG"), (3, "OK"), (7, "OK"), (8, "NG"), (4, "NG"), (9, "OK"), (5, "OK"), (10, "OK"), (6, "NG")], start=1)]
    + [dict(id=f"PRD-{10 + j:06d}", ins=f"INS-{i:06d}", ver="v2", label=label, minutes=200 + j)
       for j, (i, label) in enumerate([(11, "OK"), (12, "NG"), (11, "OK"), (7, "OK")], start=1)]
    + [dict(id="PRD-000015", ins="INS-000001", ver="v3", label="NG", minutes=300),
       dict(id="PRD-000016", ins="INS-000007", ver="v3", label="NG", minutes=300)]
)


def seed_read_only_data(session) -> None:
    """검사 12건(1~6 NG, 7~12 OK), 모델 v1(archived)·v2(production, 승인됨)·v3(candidate), 판정 16건."""
    from backend.repository import InspectionRepository, ModelVersionRepository, Stage

    repo = InspectionRepository(session)
    for n in range(1, 13):
        ng = n <= 6
        repo.add(make_inspection(n, true_label="NG" if ng else "OK", missing_parts=[2] if ng else []))
    models = ModelVersionRepository(session)
    models.register(make_metrics("v1", created_minutes=0), stage=Stage.ARCHIVED)
    models.register(make_metrics("v2", created_minutes=1), stage=Stage.PRODUCTION, approved_by="admin", approved_at=at(5))
    models.register(make_metrics("v3", created_minutes=2), stage=Stage.CANDIDATE)
    session.flush()
    insert_predictions(session.connection(), PREDICTION_SEED)


# ---------- 4-C: 가짜 모델 로더 (torch 를 가져오지 않는다) ----------
import threading


class FakeLoadError(RuntimeError):
    """가짜 모델을 불러오지 못함 (parts.predictor.ModelLoadError 대신)."""


class FakeParts:
    """InferenceService(parts_loader=...) 에 넣는 가짜. Predictor 가 하는 일 중 백엔드가 쓰는 것만 흉내 낸다.

    loads        불러온 버전을 순서대로 기록 (파일을 몇 번 읽었는지 센다)
    broken       이 버전은 불러오기에 실패한다
    thresholds   버전별 임계값 (없으면 0.5)
    on_load      load 안에서 부르는 함수(버전을 받는다): 읽는 도중에 일어나는 일을 시험한다
    predict_gate 설정하면 predict 가 이 Event 가 풀릴 때까지 기다린다 (판정 도중 모델 교체를 시험한다)
    """

    def __init__(self, thresholds=None):
        self.loads = []
        self.broken = set()
        self.thresholds = dict(thresholds or {"v1": 0.5, "v2": 0.7})
        self.on_load = None
        self.predict_gate = None
        self.predict_started = threading.Event()
        outer = self

        class Predictor:
            def __init__(self, root=None):
                self._loaded = None

            def load(self, version):
                outer.loads.append(version)
                if outer.on_load is not None:
                    outer.on_load(version)
                if version in outer.broken:
                    raise FakeLoadError(f"모델 {version} 을 불러오지 못했습니다")
                self._loaded = (version, outer.thresholds.get(version, 0.5))

            @property
            def version(self):
                return None if self._loaded is None else self._loaded[0]

            @property
            def threshold(self):
                return self._loaded[1]

            def predict(self, image):
                version, _ = self._loaded  # 시작할 때 쓸 모델을 정한다 (끝날 때까지 이 모델)
                outer.predict_started.set()
                if outer.predict_gate is not None:
                    outer.predict_gate.wait(5)
                return make_output("OK", 0.1, version)

        self.Predictor = Predictor

    def __call__(self):
        return self.Predictor, FakeLoadError
