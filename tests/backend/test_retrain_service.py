"""재학습: 저장소 보강(학습용 조회, pool 편입, 실행 중 조회, 재시작 정리)과 RetrainService (가짜 학습기 주입, torch 없음)."""
import pytest
from sqlalchemy import select

from backend.models import InspectionRow
from backend.repository import InspectionRepository, RunStatus, RunTrigger, TrainingRunRepository
from tests.backend.helpers import at, make_inspection


# ---------- 저장소 보강 ----------

def add_inspections(session, splits):
    repo = InspectionRepository(session)
    for n, split in enumerate(splits, start=1):
        repo.add(make_inspection(n, data_split=split))
    return repo


def splits_of(session):
    return {r.inspection_id: r.data_split for r in session.scalars(select(InspectionRow))}


def test_list_for_training_returns_only_train_and_pool_never_test_sets(session):
    repo = add_inspections(session, ["train", "pool", "test_normal", "test_drift", "pool"])
    got = repo.list_for_training()
    assert [i.inspection_id for i in got] == ["INS-000001", "INS-000002", "INS-000005"]
    assert {i.data_split.value for i in got} == {"train", "pool"}


def test_merge_pool_into_train_changes_only_the_given_pool_ids(session):
    repo = add_inspections(session, ["train", "pool", "pool", "test_normal", "pool"])
    changed = repo.merge_pool_into_train(["INS-000002", "INS-000001", "INS-000004", "INS-000999"])
    assert changed == 1  # pool 인 INS-000002 만 (train·테스트셋·없는 ID 는 건드리지 않는다)
    assert splits_of(session) == {"INS-000001": "train", "INS-000002": "train", "INS-000003": "pool",
                                  "INS-000004": "test_normal", "INS-000005": "pool"}
    assert repo.merge_pool_into_train([]) == 0


def test_merge_pool_into_train_handles_more_ids_than_one_chunk(session):
    repo = InspectionRepository(session)
    for n in range(1, 502):
        repo.add(make_inspection(n, data_split="pool"))
    assert repo.merge_pool_into_train([f"INS-{n:06d}" for n in range(1, 502)]) == 501
    assert repo.count(data_split="pool") == 0 and repo.count(data_split="train") == 501


def test_training_run_active_count_and_restart_cleanup(session):
    runs = TrainingRunRepository(session)
    assert runs.active() is None and runs.count() == 0
    done = runs.add("manual")
    runs.update(done.run_id, status="succeeded", finished_at=at(1))
    assert runs.active() is None
    queued, running = runs.add("manual"), runs.add("manual")
    runs.update(running.run_id, status="running", started_at=at(2), step="train")
    assert runs.active().run_id == queued.run_id and runs.count() == 3 and runs.count(status="queued") == 1
    assert runs.fail_unfinished("서버가 재시작되어 중단되었습니다", at(3)) == 2
    assert runs.active() is None and runs.get(done.run_id).status is RunStatus.SUCCEEDED
    cleaned = runs.get(running.run_id)
    assert (cleaned.status, cleaned.error_message, cleaned.finished_at, cleaned.started_at) == (
        RunStatus.FAILED, "서버가 재시작되어 중단되었습니다", at(3), at(2))
    assert runs.get(queued.run_id).started_at is None  # queued 였던 기록은 시작 시각이 없는 채로 failed
    assert runs.fail_unfinished("다시", at(4)) == 0
    with pytest.raises(ValueError):
        runs.fail_unfinished("  ", at(4))


# ---------- RetrainService (가짜 학습기, 스레드는 wait() 로 기다린다) ----------

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

import backend.retrain as retrain_module
from backend.database import session_scope
from backend.repository import (
    ApprovalRepository,
    ApprovalStatus,
    AuditLogRepository,
    ModelVersionRepository,
    RunStep,
    Stage,
)
from backend.retrain import (
    BUSY,
    INTERRUPTED,
    NO_POOL,
    DatasetUnavailableError,
    NoPoolDataError,
    RetrainBusyError,
    RetrainService,
)
from tests.backend.helpers import make_metrics, make_settings


def drift_metrics(version, accuracy, number=2):
    return make_metrics(version, created_minutes=number, drift={
        "accuracy": accuracy, "recall": 1.0, "precision": 0.9, "f1": 0.95, "confusion_matrix": [[385, 15], [0, 100]], "n": 500})


class FakeTrainer:
    """parts.predictor.train 과 같은 인자를 받는 가짜 학습기. 폴더·파일을 만들고 지표를 돌려준다 (torch 없음)."""

    def __init__(self, *, accuracy=0.97, error=None, hook=None, gate: threading.Event = None):
        self.accuracy, self.error, self.hook, self.gate = accuracy, error, hook, gate
        self.calls = []

    def __call__(self, images, labels, *, version, eval_normal, eval_drift, snapshot, root):
        self.calls.append(SimpleNamespace(images=images, labels=labels, version=version, eval_normal=eval_normal,
                                          eval_drift=eval_drift, snapshot=snapshot, root=root))
        if self.hook:
            self.hook(version)
        if self.gate is not None:
            assert self.gate.wait(10), "시험이 학습기를 풀어 주지 않았습니다"
        if self.error:
            raise self.error
        folder = Path(root) / "models" / version
        folder.mkdir(parents=True)
        (folder / "model.pt").write_bytes(b"fake")
        metrics = drift_metrics(version, self.accuracy, number=int(version[1:]))
        metrics.train_data_snapshot = {**snapshot, "seed": 42}
        (folder / "metrics.json").write_text("{}", encoding="utf-8")
        return metrics


def fake_fixed_loader(root):
    rng = np.random.default_rng(7)

    def part(n, labels):
        return SimpleNamespace(images=rng.integers(0, 90, size=(n, 128, 128), dtype=np.uint8), labels=labels)

    return {"train": part(4, ["OK", "OK", "NG", "OK"]), "test_normal": part(2, ["OK", "NG"]), "test_drift": part(2, ["OK", "NG"])}


@pytest.fixture
def world(tmp_path, factory):
    """v1 production, 고정 데이터셋(가짜 manifest), 서비스를 만드는 도우미."""
    (tmp_path / "data" / "dataset").mkdir(parents=True)
    for name in ("manifest.jsonl", "dataset_info.json"):
        (tmp_path / "data" / "dataset" / name).write_text("", encoding="utf-8")
    with session_scope(factory) as s:
        ModelVersionRepository(s).register(make_metrics("v1"), stage=Stage.PRODUCTION)
    env = SimpleNamespace(root=tmp_path, factory=factory, next_inspection=1)

    def add_inspection(split="pool", label="NG", pixel=None):
        n = env.next_inspection
        env.next_inspection += 1
        path = f"data/{'pool' if split == 'pool' else split}/INS-{n:06d}.png"
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((128, 128), pixel if pixel is not None else 100 + n, dtype=np.uint8)).save(tmp_path / path)
        with session_scope(factory) as s:
            InspectionRepository(s).add(make_inspection(n, data_split=split, true_label=label, image_path=path,
                                                         missing_parts=[2] if label == "NG" else []))
        return f"INS-{n:06d}"

    def service(trainer, **kwargs):
        return RetrainService(make_settings(root=str(tmp_path)), factory, trainer=trainer, fixed_loader=kwargs.pop("fixed_loader", fake_fixed_loader), **kwargs)

    def go(svc):
        with session_scope(factory) as s:
            run = svc.start(s, RunTrigger.MANUAL)
        assert svc.wait(20)
        return run.run_id

    def run(run_id):
        with session_scope(factory) as s:
            return TrainingRunRepository(s).get(run_id)

    def model(version):
        with session_scope(factory) as s:
            return ModelVersionRepository(s).get(version)

    def splits():
        with session_scope(factory) as s:
            return splits_of(s)

    env.add, env.service, env.go, env.run, env.model, env.splits = add_inspection, service, go, run, model, splits
    return env


def test_successful_run_walks_the_steps_and_registers_a_passing_candidate(world):
    for _ in range(3):
        world.add("pool")
    seen = {}

    def loader(root):
        seen["data"] = world.run("RUN-000001")  # data 단계에서는 이미 running
        return fake_fixed_loader(root)

    def hook(version):
        seen["train"] = world.run("RUN-000001")

    trainer = FakeTrainer(accuracy=0.97, hook=hook)
    run_id = world.go(world.service(trainer, fixed_loader=loader))
    assert (seen["data"].status, seen["data"].step) == (RunStatus.RUNNING, RunStep.DATA) and seen["data"].started_at is not None
    assert (seen["train"].status, seen["train"].step, seen["train"].finished_at) == (RunStatus.RUNNING, RunStep.TRAIN, None)
    done = world.run(run_id)
    assert (done.status, done.step, done.result_version, done.error_message) == (RunStatus.SUCCEEDED, RunStep.GATE, "v2", None)
    assert done.started_at is not None and done.finished_at >= done.started_at
    assert done.gate_result["passed"] is True and done.gate_result["baseline_version"] == "v1" and len(done.gate_result["checks"]) == 3
    v2 = world.model("v2")
    assert v2.stage is Stage.STAGING and v2.train_data_snapshot == {"train": 4, "pool": 3, "seed": 42}
    call = trainer.calls[0]
    assert call.version == "v2" and len(call.images) == len(call.labels) == 7 == call.snapshot["train"] + call.snapshot["pool"]
    assert call.labels == ["OK", "OK", "NG", "OK", "NG", "NG", "NG"] and call.root == world.root
    with session_scope(world.factory) as s:
        approval = ApprovalRepository(s).list(status="pending")[0]
        assert (approval.model_version, approval.requested_by) == ("v2", "system")
        assert [x.action.value for x in AuditLogRepository(s).list()] == ["gate_pass"]


def test_pool_is_merged_into_train_even_when_the_gate_fails(world):
    ids = [world.add("pool") for _ in range(2)]
    run_id = world.go(world.service(FakeTrainer(accuracy=0.5)))
    done = world.run(run_id)
    assert done.status is RunStatus.SUCCEEDED and done.gate_result["passed"] is False  # 게이트 미통과도 succeeded (계약 7장)
    assert world.model("v2").stage is Stage.REJECTED and done.result_version == "v2"
    with session_scope(world.factory) as s:
        assert ApprovalRepository(s).count() == 0  # 승인 요청은 만들지 않는다 (e2e X1)
        assert [x.action.value for x in AuditLogRepository(s).list()] == ["gate_fail"]
    assert all(world.splits()[i] == "train" for i in ids)  # 그래도 편입된다


def test_only_the_ids_read_in_the_data_step_are_merged_new_pool_rows_stay_pool(world):
    first = [world.add("pool") for _ in range(2)]
    late = []

    def hook(version):  # 학습하는 도중에 새 pool 이 생긴다
        late.append(world.add("pool"))

    world.go(world.service(FakeTrainer(hook=hook)))
    splits = world.splits()
    assert all(splits[i] == "train" for i in first) and splits[late[0]] == "pool"
    assert world.model("v2").train_data_snapshot["pool"] == 2  # 늦게 생긴 행은 이번 snapshot 에 들어가지 않는다


def test_failure_during_training_marks_failed_and_merges_nothing(world):
    ids = [world.add("pool") for _ in range(2)]
    run_id = world.go(world.service(FakeTrainer(error=RuntimeError("학습 중 터짐"))))
    done = world.run(run_id)
    assert (done.status, done.step, done.result_version) == (RunStatus.FAILED, RunStep.TRAIN, None)
    assert done.error_message == "학습 중 터짐" and done.finished_at is not None and done.gate_result is None
    assert all(world.splits()[i] == "pool" for i in ids)  # failed 면 편입하지 않는다
    assert world.model("v2") is None
    with session_scope(world.factory) as s:
        assert AuditLogRepository(s).list() == [] and ApprovalRepository(s).count() == 0


def test_failure_while_reading_data_marks_failed(world):
    ids = [world.add("pool") for _ in range(2)]
    (world.root / "data" / "pool" / f"{ids[0]}.png").unlink()
    trainer = FakeTrainer()
    done = world.run(world.go(world.service(trainer)))
    assert done.status is RunStatus.FAILED and done.step is RunStep.DATA and ids[0] in done.error_message
    assert trainer.calls == [] and all(world.splits()[i] == "pool" for i in ids)


def test_failure_in_the_final_transaction_undoes_registration_gate_and_merge(world, monkeypatch):
    ids = [world.add("pool") for _ in range(2)]

    def broken(*args, **kwargs):
        raise RuntimeError("게이트 기록 실패")

    monkeypatch.setattr(retrain_module, "submit_candidate", broken)
    done = world.run(world.go(world.service(FakeTrainer())))
    assert done.status is RunStatus.FAILED and done.step is RunStep.GATE and done.error_message == "게이트 기록 실패"
    assert world.model("v2") is None  # 후보 등록도 되돌려졌다
    assert all(world.splits()[i] == "pool" for i in ids)  # 편입도 일어나지 않았다
    assert (world.root / "models" / "v2").is_dir()  # 만들다 만 모델 폴더는 지우지 않는다


def test_first_model_without_production_gets_the_no_gate_result(world):
    with session_scope(world.factory) as s:
        ModelVersionRepository(s).archive_production()
    world.add("pool")
    done = world.run(world.go(world.service(FakeTrainer(accuracy=0.5))))
    assert done.gate_result == {"baseline_version": None, "passed": True, "checks": []}
    assert world.model("v2").stage is Stage.STAGING  # 게이트는 면제지만 사람 승인은 필요하다


def test_second_start_while_running_is_busy_and_creates_nothing(world):
    world.add("pool")
    release = threading.Event()
    svc = world.service(FakeTrainer(gate=release))
    with session_scope(world.factory) as s:
        first = svc.start(s, RunTrigger.MANUAL)
    try:
        with session_scope(world.factory) as s:
            with pytest.raises(RetrainBusyError, match=BUSY):
                svc.start(s, RunTrigger.MANUAL)
            assert TrainingRunRepository(s).count() == 1
    finally:
        release.set()
    assert svc.wait(20) and world.run(first.run_id).status is RunStatus.SUCCEEDED


def test_a_leftover_running_row_blocks_a_new_run_until_restart_cleanup(world):
    world.add("pool")
    with session_scope(world.factory) as s:
        TrainingRunRepository(s).add("manual")  # queued 로 남은 기록 (스레드는 없음)
    svc = world.service(FakeTrainer())
    with session_scope(world.factory) as s:
        with pytest.raises(RetrainBusyError):
            svc.start(s, RunTrigger.MANUAL)
    assert svc.mark_interrupted() == 1
    assert world.run("RUN-000001").error_message == INTERRUPTED
    assert world.run(world.go(svc)).status is RunStatus.SUCCEEDED


def test_start_without_new_pool_data_or_dataset_creates_nothing(world, tmp_path):
    svc = world.service(FakeTrainer())
    with session_scope(world.factory) as s:
        with pytest.raises(NoPoolDataError, match=NO_POOL):
            svc.start(s, RunTrigger.MANUAL)
        assert TrainingRunRepository(s).count() == 0
    world.add("train")  # train 만 있어도 pool 이 아니다
    world.add("test_normal")
    with session_scope(world.factory) as s:
        with pytest.raises(NoPoolDataError):
            svc.start(s, RunTrigger.MANUAL)
    (world.root / "data" / "dataset" / "manifest.jsonl").unlink()
    with session_scope(world.factory) as s:
        with pytest.raises(DatasetUnavailableError):
            svc.start(s, RunTrigger.MANUAL)
        assert TrainingRunRepository(s).count() == 0


def test_version_number_counts_db_versions_and_unregistered_model_folders(world):
    with session_scope(world.factory) as s:
        ModelVersionRepository(s).register(make_metrics("v3", created_minutes=3), stage=Stage.ARCHIVED)
    (world.root / "models" / "v5").mkdir(parents=True)  # DB 에 없는 폴더
    (world.root / "models" / "notes").mkdir()  # v숫자가 아닌 폴더는 무시
    world.add("pool")
    trainer = FakeTrainer()
    done = world.run(world.go(world.service(trainer)))
    assert trainer.calls[0].version == "v6" and done.result_version == "v6"


def test_test_sets_never_enter_the_training_data(world):
    world.add("pool", pixel=111)
    world.add("test_normal", pixel=222)
    world.add("test_drift", pixel=233)
    world.add("train", pixel=144)
    trainer = FakeTrainer()
    world.go(world.service(trainer))
    images = trainer.calls[0].images
    assert len(images) == 4 + 1 + 1 and not np.any(images == 222) and not np.any(images == 233)
    assert trainer.calls[0].snapshot == {"train": 5, "pool": 1}  # 고정 train 4 + DB train 1


def test_mark_interrupted_only_touches_unfinished_runs(world):
    with session_scope(world.factory) as s:
        runs = TrainingRunRepository(s)
        done = runs.add("manual")
        runs.update(done.run_id, status="succeeded", finished_at=at(1))
        running = runs.add("manual")
        runs.update(running.run_id, status="running", started_at=at(2), step="train")
    svc = world.service(FakeTrainer())
    assert svc.mark_interrupted() == 1 and svc.mark_interrupted() == 0
    assert world.run(done.run_id).status is RunStatus.SUCCEEDED
    cleaned = world.run(running.run_id)
    assert (cleaned.status, cleaned.error_message, cleaned.step) == (RunStatus.FAILED, INTERRUPTED, RunStep.TRAIN)


def test_real_training_runs_end_to_end_with_the_real_train_function(world):
    """parts.predictor.train() 의 호출 방식이 바뀌면 가짜 학습기 시험은 모두 통과해도 실제로는 깨진다. 이 시험이 그것을 지킨다."""
    for _ in range(3):
        world.add("pool")
    svc = RetrainService(make_settings(root=str(world.root)), world.factory, fixed_loader=fake_fixed_loader)  # 실제 학습기
    done = world.run(world.go(svc))
    assert done.status is RunStatus.SUCCEEDED and done.result_version == "v2"
    assert (world.root / "models" / "v2" / "model.pt").is_file() and (world.root / "models" / "v2" / "metrics.json").is_file()
    assert world.model("v2").train_data_snapshot["pool"] == 3
