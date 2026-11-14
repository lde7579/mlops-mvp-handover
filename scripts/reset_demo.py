"""시연 초기화: 시연 중에 쌓인 데이터를 지우고 "v1 만 운영 중인 처음 상태"로 되돌린다.

실행 (서버를 끈 상태에서):
  python -m scripts.reset_demo            미리보기. 지울 것을 보여 주기만 하고 아무것도 바꾸지 않는다
  python -m scripts.reset_demo --yes      실제로 초기화한다
  python -m scripts.reset_demo --check    지금 상태가 "처음 상태"인지 읽기만 해서 확인한다 (파일을 만들거나 바꾸지 않는다)

지우는 것: DB 의 검사·판정·승인·학습·배포·감사 로그 행, model_version 전부(v1 은 models/v1 에서 다시 등록),
          data/pool 의 INS-숫자.png, models/ 의 v2 이상 폴더.
남기는 것: models/v1 (읽기만 한다), data/dataset, data/samples, data/secret.key 등 그 밖의 모든 파일.

순서와 이유 (docs 계획서 7절): DB(한 트랜잭션) -> PNG -> 모델 폴더.
- DB 가 먼저 비면 남은 PNG 는 주인 없는 파일일 뿐이라 무해하다. 반대로 PNG 만 지워지고 DB 가 남으면 재학습과 이미지 조회가 깨진다.
- 중간에 실패해도 같은 명령(--yes)을 다시 실행하면 이어서 끝난다.
서버가 켜져 있으면 거부한다 (서버 메모리의 시뮬레이터·번호·모델이 DB 와 어긋나기 때문). 서버를 끄고 실행한다.
"""
import argparse
import hashlib
import json
import re
import shutil
import socket
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sqlalchemy import delete

from backend import models as M
from backend.database import create_db_engine, init_db, make_session_factory, session_scope
from backend.registry import models_root
from backend.repository import ModelVersionRepository, Stage
from backend.settings import PROJECT_ROOT, Settings, load_settings
from common.schemas import ModelMetrics

KEEP_VERSION = "v1"
DEFAULT_PORT = 8000
PNG_NAME = re.compile(r"^INS-[0-9]+\.png$")
MODEL_DIR_NAME = re.compile(r"^v[0-9]+$")
EMPTY_TABLES = ("inspection", "prediction", "approval_request", "training_run", "deployment", "audit_log")
# 지우는 순서: 자식(외래키로 참조하는 쪽) 먼저
DELETE_ORDER = (M.PredictionRow, M.AuditLogRow, M.DeploymentRow, M.ApprovalRequestRow, M.TrainingRunRow,
                M.InspectionRow, M.ModelVersionRow)
OTHER_SPLIT_DIRS = ("train", "test_normal", "test_drift")  # 수집기가 번호를 정할 때 보는 폴더 (지우지 않고 알리기만)


class ResetError(Exception):
    """초기화를 시작할 수 없다 (아무것도 바꾸지 않았다)."""


@dataclass
class State:
    db_exists: bool = False
    counts: dict = field(default_factory=dict)          # 표 이름 -> 행 수 (표가 없으면 0)
    model_rows: list = field(default_factory=list)      # [(version, stage, metrics dict)]
    pool_pngs: list = field(default_factory=list)       # 지울 수 있는 PNG 경로
    pool_kept: list = field(default_factory=list)       # 지우지 않을 항목 이름 (INS-숫자.png 가 아님 등)
    model_dirs: list = field(default_factory=list)      # 지울 모델 폴더 경로 (v1 제외)
    model_dirs_kept: list = field(default_factory=list)
    v1_ok: bool = False
    v1_hashes: dict = field(default_factory=dict)
    other_split_files: int = 0


# ---------- 읽기 (아무것도 만들지 않는다) ----------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_db(db_path: Path, state: State) -> None:
    """DB 를 읽기 전용으로 연다. 파일이 없거나 표가 없으면 0건으로 센다."""
    if not db_path.is_file():
        return
    state.db_exists = True
    # WAL 방식 DB 는 읽기만 해도 -wal, -shm 파일이 생길 수 있다. 두 파일이 모두 없으면 아직 WAL 에 쌓인 내용도 없으므로
    # immutable=1 로 열어 파일을 하나도 만들지 않고 읽는다. 하나라도 있으면(서버가 쓰던 DB) 일반 읽기 전용으로 열어 WAL 내용까지 본다.
    wal_files = (db_path.with_name(db_path.name + "-wal"), db_path.with_name(db_path.name + "-shm"))
    flag = "mode=ro" if any(p.exists() for p in wal_files) else "immutable=1"
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?{flag}", uri=True, timeout=1.0)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for name in EMPTY_TABLES:
            state.counts[name] = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] if name in tables else 0
        if "model_version" in tables:
            state.model_rows = [(v, s, json.loads(m)) for v, s, m in
                                conn.execute("SELECT version, stage, metrics FROM model_version ORDER BY version")]
    finally:
        conn.close()


def _is_inside(path: Path, base: Path) -> bool:
    """path 가 링크가 아니고, 실제 위치의 바로 윗 폴더가 base 이다."""
    return not path.is_symlink() and path.resolve().parent == base.resolve()


def read_state(root, db_path) -> State:
    root, db_path = Path(root), Path(db_path)
    state = State()
    _read_db(db_path, state)

    pool = root / "data" / "pool"
    if pool.is_dir():
        for p in sorted(pool.iterdir(), key=lambda q: q.name):
            if p.is_file() and PNG_NAME.match(p.name) and _is_inside(p, pool):
                state.pool_pngs.append(p)
            else:
                state.pool_kept.append(p.name)
    for name in OTHER_SPLIT_DIRS:
        folder = root / "data" / name
        if folder.is_dir():
            state.other_split_files += sum(1 for p in folder.iterdir() if PNG_NAME.match(p.name))

    models = models_root(root)
    if models.is_dir():
        for p in sorted(models.iterdir(), key=lambda q: q.name):
            if p.name == KEEP_VERSION:
                continue
            if p.is_dir() and MODEL_DIR_NAME.match(p.name) and _is_inside(p, models):
                state.model_dirs.append(p)
            else:
                state.model_dirs_kept.append(p.name)
    v1 = models / KEEP_VERSION
    files = {name: v1 / name for name in ("model.pt", "metrics.json")}
    state.v1_ok = v1.is_dir() and all(p.is_file() for p in files.values())
    if state.v1_ok:
        state.v1_hashes = {name: _sha256(p) for name, p in files.items()}
    return state


def read_v1_metrics(root) -> ModelMetrics:
    path = models_root(Path(root)) / KEEP_VERSION / "metrics.json"
    try:
        metrics = ModelMetrics.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception as e:
        raise ResetError(f"models/{KEEP_VERSION}/metrics.json 을 읽을 수 없어 시작하지 않았습니다: {e}") from e
    if metrics.version != KEEP_VERSION:
        raise ResetError(f"models/{KEEP_VERSION}/metrics.json 의 version({metrics.version})이 폴더 이름과 다릅니다")
    return metrics


def problems(state: State, expected: Optional[ModelMetrics]) -> list:
    """"처음 상태"와 다른 점. 비어 있으면 합격."""
    out = [f"{name} 행이 {n}건 남아 있음" for name, n in state.counts.items() if n]
    rows = state.model_rows
    if len(rows) != 1 or rows[0][0] != KEEP_VERSION or rows[0][1] != Stage.PRODUCTION.value:
        shown = ", ".join(f"{v}({s})" for v, s, _ in rows) or "없음"
        out.append(f"model_version 이 {KEEP_VERSION}(production) 1개가 아님: {shown}")
    elif expected is not None and rows[0][2] != expected.to_dict():
        out.append(f"DB 의 {KEEP_VERSION} 지표가 models/{KEEP_VERSION}/metrics.json 과 다름")
    if state.pool_pngs:
        out.append(f"data/pool 에 PNG 가 {len(state.pool_pngs)}장 남아 있음")
    if state.model_dirs:
        out.append("models/ 에 남은 폴더: " + ", ".join(p.name for p in state.model_dirs))
    if not state.v1_ok:
        out.append(f"models/{KEEP_VERSION} 에 model.pt 또는 metrics.json 이 없음")
    return out


# ---------- 안전장치 ----------

def server_listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def db_is_locked(db_path: Path) -> bool:
    """다른 프로세스가 DB 에 쓰는 중이면 True (쓰기 잠금을 잡아 보고 바로 놓는다)."""
    if not db_path.is_file():
        return False
    conn = sqlite3.connect(str(db_path), timeout=0.3, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("ROLLBACK")
        return False
    except sqlite3.OperationalError:
        return True
    finally:
        conn.close()


# ---------- 지우기 ----------

def reset_database(settings: Settings, v1_metrics: ModelMetrics) -> None:
    """한 트랜잭션: 전부 지우고 v1 을 production 으로 다시 등록한다. 실패하면 아무것도 바뀌지 않는다."""
    engine = create_db_engine(settings.db_path)
    try:
        init_db(engine)  # 옛 DB 에 없는 표를 만든다
        with session_scope(make_session_factory(engine)) as session:
            for row_class in DELETE_ORDER:
                session.execute(delete(row_class))
            ModelVersionRepository(session).register(v1_metrics, stage=Stage.PRODUCTION)
    finally:
        engine.dispose()


def remove_pool_pngs(root, files, log=print) -> list:
    """INS-숫자.png 만 지운다. 실패한 항목은 (이름, 이유) 목록으로 돌려준다."""
    base, failed = Path(root) / "data" / "pool", []
    for p in files:
        try:
            if not (PNG_NAME.match(p.name) and p.is_file() and _is_inside(p, base)):
                raise OSError("지우면 안 되는 경로라 건너뜀")
            p.unlink()
        except OSError as e:
            failed.append((p.name, str(e)))
    return failed


def remove_model_dirs(root, dirs, log=print) -> list:
    """v숫자(v1 제외) 폴더만 지운다."""
    base, failed = models_root(Path(root)), []
    for p in dirs:
        try:
            if p.name == KEEP_VERSION or not (MODEL_DIR_NAME.match(p.name) and p.is_dir() and _is_inside(p, base)):
                raise OSError("지우면 안 되는 경로라 건너뜀")
            shutil.rmtree(p)
        except OSError as e:
            failed.append((p.name, str(e)))
    return failed


def checkpoint(db_path: Path) -> None:
    """WAL 파일의 내용을 DB 본체로 합친다 (실패해도 초기화 결과에는 영향이 없다)."""
    if not db_path.is_file():
        return
    try:
        conn = sqlite3.connect(str(db_path), timeout=1.0)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()
    except sqlite3.Error:
        pass


# ---------- 출력 ----------

def describe(state: State, out) -> None:
    db_line = ", ".join(f"{k} {v}" for k, v in state.counts.items()) if state.db_exists else "DB 파일 없음"
    out(f"DB 행: {db_line}")
    out("model_version: " + (", ".join(f"{v}({s})" for v, s, _ in state.model_rows) or "없음"))
    out(f"data/pool PNG: {len(state.pool_pngs)}장" + (f" ({state.pool_pngs[0].name} ~ {state.pool_pngs[-1].name})" if state.pool_pngs else ""))
    out("지울 모델 폴더: " + (", ".join(p.name for p in state.model_dirs) or "없음"))
    if state.pool_kept:
        out(f"지우지 않는 data/pool 항목: {', '.join(state.pool_kept)}")
    if state.model_dirs_kept:
        out(f"지우지 않는 models 항목: {', '.join(state.model_dirs_kept)}")
    if state.other_split_files:
        out(f"주의: data/train·test_*  폴더에 INS PNG 가 {state.other_split_files}장 있습니다 (지우지 않음, 검사 번호에는 반영됨)")
    if state.v1_ok:
        out(f"유지: models/{KEEP_VERSION} (model.pt {state.v1_hashes['model.pt'][:8]}…, metrics.json {state.v1_hashes['metrics.json'][:8]}…)")


def do_check(root, db_path, out) -> int:
    state = read_state(root, db_path)
    try:
        expected = read_v1_metrics(root) if state.v1_ok else None
    except ResetError as e:
        out(f"주의: {e}")
        expected = None
    describe(state, out)
    found = problems(state, expected)
    if found:
        out("불합격 - 처음 상태가 아닙니다:")
        for line in found:
            out(f"  - {line}")
        out("처음 상태로 되돌리려면 서버를 끄고: python -m scripts.reset_demo --yes")
        return 1
    out("합격 - 처음 상태입니다 (검사·판정·승인·학습·배포·감사 0건, v1 production, pool PNG 0장).")
    return 0


# ---------- 실행 ----------

def main(argv=None, root=PROJECT_ROOT, settings: Optional[Settings] = None, out=print, port_probe=server_listening) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.reset_demo", description="시연 데이터를 지우고 v1 만 운영 중인 처음 상태로 되돌린다")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--yes", action="store_true", help="실제로 초기화한다 (없으면 미리보기만)")
    mode.add_argument("--check", action="store_true", help="처음 상태인지 읽기만 해서 확인한다")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"서버가 켜져 있는지 확인할 포트 (기본 {DEFAULT_PORT})")
    args = parser.parse_args(argv)

    root = Path(root)
    # 비밀키는 쓰지 않는다 (data/secret.key 를 만들지 않도록 더미 값을 준다)
    settings = settings if settings is not None else load_settings(root=root, secret_key="x" * 32)
    db_path = Path(settings.db_path)

    if args.check:
        return do_check(root, db_path, out)

    state = read_state(root, db_path)
    out("== 시연 초기화 " + ("(실행)" if args.yes else "(미리보기: 아무것도 바꾸지 않습니다)") + " ==")
    describe(state, out)
    running = port_probe(args.port)
    if not args.yes:
        if running:
            out(f"경고: 127.0.0.1:{args.port} 에 응답하는 프로그램이 있습니다. 실행하려면 서버를 먼저 끄세요.")
        out("실제로 지우려면: python -m scripts.reset_demo --yes")
        return 0

    try:
        if running:
            raise ResetError(f"127.0.0.1:{args.port} 에 응답하는 프로그램이 있습니다. 서버를 먼저 끄세요. 아무것도 지우지 않았습니다")
        if db_is_locked(db_path):
            raise ResetError("DB 를 다른 프로그램이 쓰고 있습니다. 서버를 끄고 다시 실행하세요. 아무것도 지우지 않았습니다")
        v1_metrics = read_v1_metrics(root)
        out("--- 지우기 시작 ---")
        reset_database(settings, v1_metrics)
        out(f"DB 초기화 완료: 모든 행 삭제, {KEEP_VERSION}(production) 다시 등록")
    except ResetError as e:
        print(f"실패: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # DB 단계 실패: 트랜잭션이 되돌려졌으므로 아무것도 바뀌지 않았다
        print(f"실패: DB 를 초기화하지 못했습니다 (아무것도 바뀌지 않았습니다): {e}", file=sys.stderr)
        return 1

    failed = remove_pool_pngs(root, state.pool_pngs, out)
    out(f"data/pool PNG {len(state.pool_pngs) - len(failed)}장 삭제")
    failed += remove_model_dirs(root, state.model_dirs, out)
    out("모델 폴더 삭제: " + (", ".join(p.name for p in state.model_dirs) or "없음"))
    checkpoint(db_path)
    if failed:
        for name, reason in failed:
            print(f"실패: {name} 을(를) 지우지 못했습니다: {reason}", file=sys.stderr)
        print("같은 명령(python -m scripts.reset_demo --yes)을 다시 실행하면 이어서 지웁니다.", file=sys.stderr)
        return 1
    return do_check(root, db_path, out)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
