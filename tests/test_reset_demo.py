"""scripts/reset_demo 테스트. 전부 임시 폴더(tmp_path) 안에서만 한다. 실제 data/ 와 models/ 는 건드리지 않는다.

루트를 tmp_path 로 넘기고, DB 도 tmp_path/data/mlops.db 로 잡는다 (load_settings(root=tmp_path)).
"""
import hashlib
import json
import shutil
import socket
from pathlib import Path
import sqlite3
from datetime import datetime, timezone

import pytest

import scripts.reset_demo as reset_demo
from backend import models as M
from backend.database import create_db_engine, init_db, make_session_factory, session_scope
from backend.generation import GenerationService
from backend.repository import ModelVersionRepository, Stage
from backend.settings import load_settings
from scripts.reset_demo import main
from tests.backend.helpers import make_metrics

NOW = datetime(2026, 10, 31, 9, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def root(tmp_path):
    v1 = tmp_path / "models" / "v1"
    v1.mkdir(parents=True)
    (v1 / "model.pt").write_bytes(b"fake-model-weights")
    (v1 / "metrics.json").write_text(json.dumps(make_metrics("v1").to_dict()), encoding="utf-8")
    return tmp_path


@pytest.fixture
def settings(root):
    return load_settings(root=root, secret_key="x" * 32)


def run(root, settings, *argv, probe=lambda port: False):
    """probe: 서버 확인을 가짜로 바꾼다 (윈도우는 닫힌 포트 접속 실패가 느려서). 진짜 접속 시험은 따로 있다."""
    lines = []
    code = main(list(argv), root=root, settings=settings, out=lines.append, port_probe=probe)
    return code, "\n".join(lines)


def tree(root):
    """폴더 안 모든 파일의 상대경로 -> 해시 (바뀐 것·생긴 것·사라진 것을 한 번에 비교)."""
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def counts(settings):
    conn = sqlite3.connect(f"{Path(settings.db_path).as_uri()}?immutable=1", uri=True)  # 읽기만 하고 -wal, -shm 을 만들지 않는다
    try:
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in
                ("inspection", "prediction", "model_version", "approval_request", "training_run", "deployment", "audit_log")}
    finally:
        conn.close()


def make_demo_leftovers(root, settings, pngs=3):
    """시연 한 바퀴가 끝난 상태: v1 archived, v2 production, 각 표에 행, pool PNG, models/v2 폴더."""
    engine = create_db_engine(settings.db_path)
    init_db(engine)
    with session_scope(make_session_factory(engine)) as session:
        models = ModelVersionRepository(session)
        models.register(make_metrics("v1"), stage=Stage.ARCHIVED, approved_by="admin", approved_at=NOW)
        models.register(make_metrics("v2", created_minutes=5), stage=Stage.PRODUCTION, approved_by="admin", approved_at=NOW)
        session.add(M.InspectionRow(inspection_id="INS-000001", captured_at=NOW, image_path="data/pool/INS-000001.png",
                                    true_label="NG", missing_parts=[2], scenario="normal", drift_type=None, data_split="train"))
        session.flush()
        session.add_all([
            M.PredictionRow(prediction_id="PRD-000001", inspection_id="INS-000001", model_version="v1", predicted_label="NG",
                            score=0.9, threshold=0.5, latency_ms=3, predicted_at=NOW),
            M.TrainingRunRow(run_id="RUN-000001", status="succeeded", trigger="manual", result_version="v2"),
            M.ApprovalRequestRow(approval_id="APR-000001", model_version="v2", requested_by="operator", status="approved", created_at=NOW),
            M.DeploymentRow(deployment_id="DEP-000001", version="v2", action="deploy", actor="admin", at=NOW),
            M.AuditLogRow(log_id="LOG-000001", actor="admin", role="admin", action="approve", target="v2", at=NOW),
        ])
    engine.dispose()
    pool = root / "data" / "pool"
    pool.mkdir(parents=True, exist_ok=True)
    for n in range(1, pngs + 1):
        (pool / f"INS-{n:06d}.png").write_bytes(b"png")
    (root / "models" / "v2").mkdir()
    (root / "models" / "v2" / "model.pt").write_bytes(b"v2")


def test_preview_changes_nothing_and_lists_targets(root, settings):
    make_demo_leftovers(root, settings)
    rows = counts(settings)
    before = tree(root)
    code, text = run(root, settings)
    assert code == 0
    assert "미리보기" in text and "INS-000001.png ~ INS-000003.png" in text and "v2" in text and "--yes" in text
    assert tree(root) == before and counts(settings) == rows


def test_yes_restores_the_initial_state_and_keeps_v1_bytes(root, settings):
    make_demo_leftovers(root, settings)
    (root / "data" / "dataset").mkdir()
    (root / "data" / "dataset" / "manifest.jsonl").write_text("keep", encoding="utf-8")
    v1_before = {k: v for k, v in tree(root).items() if k.startswith("models") and "v1" in k}
    code, text = run(root, settings, "--yes")
    assert code == 0 and "합격" in text
    assert "INS-000001.png ~ INS-000003.png" in text and "v2" in text  # 지우기 전에 대상 목록을 보여 준다
    assert counts(settings) == {"inspection": 0, "prediction": 0, "model_version": 1, "approval_request": 0,
                                "training_run": 0, "deployment": 0, "audit_log": 0}
    conn = sqlite3.connect(settings.db_path)
    try:
        assert conn.execute("SELECT version, stage, approved_by, approved_at FROM model_version").fetchall() == [("v1", "production", None, None)]
    finally:
        conn.close()
    after = tree(root)
    assert not [k for k in after if k.startswith("data\\pool") or k.startswith("models\\v2")]
    assert {k: v for k, v in after.items() if k.startswith("models") and "v1" in k} == v1_before
    assert (root / "data" / "dataset" / "manifest.jsonl").read_text(encoding="utf-8") == "keep"


def test_kept_entries_survive_and_result_is_pass(root, settings):
    make_demo_leftovers(root, settings, pngs=1)
    pool = root / "data" / "pool"
    (pool / "notes.txt").write_text("memo", encoding="utf-8")
    (pool / "INS-000009.png").mkdir()
    (root / "models" / "old").mkdir()
    code, text = run(root, settings, "--yes")
    assert code == 0
    assert (pool / "notes.txt").exists() and (pool / "INS-000009.png").is_dir() and (root / "models" / "old").is_dir()
    assert not (pool / "INS-000001.png").exists() and not (root / "models" / "v2").exists()
    assert "notes.txt" in text and "old" in text  # 지우지 않는 항목으로 알려 준다


def test_helpers_refuse_paths_outside_the_allowed_folders(root, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    stray_png, stray_dir = outside / "INS-000001.png", outside / "v7"
    stray_png.write_bytes(b"x")
    stray_dir.mkdir()
    (root / "models" / "v1").mkdir(exist_ok=True)
    assert [n for n, _ in reset_demo.remove_pool_pngs(root, [stray_png])] == ["INS-000001.png"]
    assert [n for n, _ in reset_demo.remove_model_dirs(root, [stray_dir, root / "models" / "v1"])] == ["v7", "v1"]
    assert stray_png.exists() and stray_dir.exists() and (root / "models" / "v1" / "model.pt").exists()


def test_running_server_refuses_yes_but_preview_only_warns(root, settings):
    make_demo_leftovers(root, settings)
    rows = counts(settings)
    before = tree(root)
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = str(server.getsockname()[1])
        real = reset_demo.server_listening
        code, text = run(root, settings, "--yes", "--port", port, probe=real)
        assert code == 1 and tree(root) == before and counts(settings) == rows
        code, text = run(root, settings, "--port", port, probe=real)
        assert code == 0 and "경고" in text and "서버" in text
    assert tree(root) == before


def test_locked_db_refuses_yes(root, settings):
    make_demo_leftovers(root, settings)
    rows = counts(settings)
    holder = sqlite3.connect(settings.db_path, isolation_level=None)
    try:
        holder.execute("BEGIN IMMEDIATE")
        code, _ = run(root, settings, "--yes")
        assert code == 1
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert counts(settings) == rows and (root / "data" / "pool" / "INS-000001.png").exists()


def test_failure_in_the_file_step_is_finished_by_running_again(root, settings, monkeypatch):
    make_demo_leftovers(root, settings)
    monkeypatch.setattr(reset_demo, "remove_model_dirs", lambda r, dirs, log=print: [("v2", "boom")])
    code, _ = run(root, settings, "--yes")
    assert code == 1
    assert counts(settings)["inspection"] == 0 and (root / "models" / "v2").exists()  # DB 는 이미 비었고 폴더만 남음
    code, text = run(root, settings, "--check")
    assert code == 1 and "v2" in text  # 불일치를 불합격으로 알린다
    monkeypatch.undo()
    code, _ = run(root, settings, "--yes")
    assert code == 0 and not (root / "models" / "v2").exists()


def test_old_schema_db_is_counted_as_zero_and_check_creates_nothing(root, settings):
    engine = create_db_engine(settings.db_path)
    for table in (M.InspectionRow, M.ModelVersionRow, M.PredictionRow):  # 옛 스키마: 승인·학습·배포·감사 표가 없다
        table.__table__.create(engine)
    with session_scope(make_session_factory(engine)) as session:
        ModelVersionRepository(session).register(make_metrics("v1"), stage=Stage.PRODUCTION)
    engine.dispose()
    before = tree(root)
    code, text = run(root, settings, "--check")
    assert code == 0 and "합격" in text
    assert tree(root) == before  # 파일을 만들거나 바꾸지 않았다 (-wal, -shm 포함)
    conn = sqlite3.connect(settings.db_path)
    try:
        assert "approval_request" not in {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    finally:
        conn.close()
    code, _ = run(root, settings)  # 미리보기도 마찬가지
    assert code == 0 and tree(root) == before
    code, _ = run(root, settings, "--yes")
    assert code == 0 and counts(settings)["audit_log"] == 0  # 실행하면 빠진 표를 만든다


def test_check_is_read_only_and_fails_on_leftovers(root, settings):
    make_demo_leftovers(root, settings)
    before = tree(root)
    code, text = run(root, settings, "--check")
    assert code == 1 and "불합격" in text and "pool" in text and "model_version" in text
    assert tree(root) == before
    assert run(root, settings, "--yes")[0] == 0
    before = tree(root)
    assert run(root, settings, "--check")[0] == 0
    assert tree(root) == before


def test_missing_v1_files_stop_before_anything_is_deleted(root, settings):
    make_demo_leftovers(root, settings)
    (root / "models" / "v1" / "metrics.json").unlink()
    rows = counts(settings)
    before = tree(root)
    code, _ = run(root, settings, "--yes")
    assert code == 1 and tree(root) == before and counts(settings) == rows


def test_no_db_file_means_zero_rows_and_yes_creates_the_initial_db(root, settings):
    code, text = run(root, settings, "--check")
    assert code == 1 and "불합격" in text
    assert run(root, settings, "--yes")[0] == 0
    assert counts(settings)["model_version"] == 1


def test_first_inspection_after_reset_is_number_one(root, settings):
    make_demo_leftovers(root, settings)
    assert run(root, settings, "--yes")[0] == 0
    (root / "parts" / "collector").mkdir(parents=True)  # 수집기 설정 파일을 임시 루트로 복사 (원본은 읽기만)
    shutil.copy(Path(__file__).resolve().parents[1] / "parts" / "collector" / "collector.toml", root / "parts" / "collector")
    engine = create_db_engine(settings.db_path)
    try:
        with session_scope(make_session_factory(engine)) as session:
            inspection = GenerationService(settings).generate(session, "normal")
    finally:
        engine.dispose()
    assert inspection.inspection_id == "INS-000001"
    assert (root / "data" / "pool" / "INS-000001.png").is_file()


def test_yes_and_check_cannot_be_combined(root, settings):
    with pytest.raises(SystemExit):
        run(root, settings, "--yes", "--check")
