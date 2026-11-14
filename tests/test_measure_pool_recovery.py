"""scripts/measure_pool_recovery.py 테스트: 학습은 가짜로 바꾸고(torch 불필요), 이미지는 아주 작게 만든다.

핵심 확인: 측정용 pool 이 실제 생성 경로(SimulatorRunner + GenerationService)와 같은 이미지를 만드는가 (해시 비교, dark 켠 구간 포함).
"""
import hashlib
import random
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from backend.database import create_db_engine, init_db, make_session_factory, session_scope
from backend.generation import GenerationService
from backend.simulator_runner import DEFAULT_DEFECT_RATE, LABEL_SEED, SimulatorRunner
from common.schemas import ClassMetrics, ModelMetrics
from scripts.make_dataset import DatasetError
from scripts.measure_pool_recovery import (
    PASS_RATE,
    RATIOS,
    REPS,
    RULE_LINES,
    SIZES,
    cell_summary,
    demo_image_seed,
    fingerprint,
    group_cells,
    image_seed_for,
    make_pool,
    md5s,
    minimum_size,
    pool_plan,
    recommend,
    run_measure,
    run_one,
    train_seed_for,
)
from tests.backend.helpers import make_settings


def md5(arr) -> str:
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


# ---------- pool 구성 ----------

def test_pool_plan_counts():
    assert pool_plan(100, 0.2) == (80, 20)
    assert pool_plan(100, 0.4) == (60, 40)
    assert pool_plan(100, 0.6) == (40, 60)
    assert pool_plan(30, 0.2) == (24, 6)
    for size in SIZES:
        for ratio in RATIOS:
            n_normal, n_dark = pool_plan(size, ratio)
            assert n_normal + n_dark == size and n_dark > 0


def test_seeds_follow_the_demo():
    assert demo_image_seed() == 43  # 설정 시드 42 + 시작 번호 1
    assert image_seed_for(0) == 43 and train_seed_for(0) == 42
    assert image_seed_for(1) == 101 and train_seed_for(1) == 43
    assert len({image_seed_for(k) for k in range(REPS)}) == REPS


def test_make_pool_is_deterministic_and_seed_dependent(tmp_path):
    a = make_pool(3, 2, 43, tmp_path / "a")
    b = make_pool(3, 2, 43, tmp_path / "b")
    c = make_pool(3, 2, 101, tmp_path / "c")
    assert a.images.shape == (5, 128, 128) and a.images.dtype == np.uint8
    assert (a.n_normal, a.n_dark) == (3, 2)
    assert md5s(a.images) == md5s(b.images) and a.labels == b.labels
    assert md5s(a.images) != md5s(c.images)


def test_make_pool_matches_real_generation_path(tmp_path):
    """SimulatorRunner(자동 생성) + GenerationService 가 정상 4장 뒤 dark 3장을 만든 이미지와 해시가 모두 같다."""
    n_normal, n_dark = 4, 3
    total = n_normal + n_dark
    real_root = tmp_path / "real"
    engine = create_db_engine(":memory:")
    init_db(engine)
    factory = make_session_factory(engine)
    service = GenerationService(make_settings(root=str(real_root)))
    made = []

    def generate(scenario, label, drift_type):
        with session_scope(factory) as session:
            made.append(service.generate(session, scenario, label=label, drift_type=drift_type))
        return made[-1]

    ticks = []
    runner = SimulatorRunner(generate, lambda inspection: None)

    def fake_wait(stop, seconds):
        ticks.append(1)
        if len(ticks) == n_normal:
            runner.set_drift(True, "dark")  # 정상 4장 뒤부터 dark (시연의 단계 3)
        return len(ticks) >= total

    runner._wait = fake_wait
    try:
        runner.start(defect_rate=DEFAULT_DEFECT_RATE, interval_seconds=0)
        runner.thread.join(30)
        assert not runner.thread.is_alive() and len(made) == total
        real = []
        for inspection in made:
            with Image.open(real_root / inspection.image_path) as png:
                real.append(np.asarray(png).copy())
    finally:
        engine.dispose()

    mine = make_pool(n_normal, n_dark, demo_image_seed(), tmp_path / "mine")
    assert [i.drift_type for i in made] == [None] * n_normal + ["dark"] * n_dark
    assert [i.true_label.value for i in made] == mine.labels
    assert [md5(im) for im in real] == [md5(im) for im in mine.images]
    # 대조군: 불량 순서 규칙이 실제와 같은지 (난수 순서를 직접 재생)
    rng = random.Random(LABEL_SEED)
    assert mine.labels == ["NG" if rng.random() < DEFAULT_DEFECT_RATE else "OK" for _ in range(total)]


# ---------- 합격 규칙 ----------

def fake_run(size, ratio, rep, passed):
    return {"size": size, "ratio": ratio, "rep": rep, "passed": passed, "kind_accuracy": {"dark": 0.9, "noise": 0.8, "shift": 0.7}, "drift_accuracy": 0.9,
            "seconds": 1.0, "n_normal": 1, "n_dark": 1, "n_ng": 0, "image_seed": 0, "train_seed": 0, "checks": {}}


def cell_runs(size, ratio, results):
    return [fake_run(size, ratio, k, ok) for k, ok in enumerate(results)]


def test_pass_rate_constant():
    assert PASS_RATE == 0.8
    assert any("80%" in line for line in RULE_LINES)


def test_cell_ok_needs_rate_and_first_rep():
    assert cell_summary(cell_runs(100, 0.2, [True, True, True, True, False]))["ok"]          # 4/5 + 1회차 통과
    assert not cell_summary(cell_runs(100, 0.2, [True, True, True, False, False]))["ok"]     # 3/5
    c = cell_summary(cell_runs(100, 0.2, [False, True, True, True, True]))                   # 4/5 이지만 1회차 불합격
    assert c["rate"] == 0.8 and not c["first_ok"] and not c["ok"]
    assert cell_summary(cell_runs(100, 0.2, [True]))["ok"]


def make_cells(table):
    """table: {비율: {크기: 칸 합격 여부}} -> group_cells 결과."""
    runs = []
    for ratio, by_size in table.items():
        for size, ok in by_size.items():
            runs += cell_runs(size, ratio, [ok] * 5)
    return group_cells(runs)


def test_minimum_size_requires_all_larger_sizes_to_pass():
    flags = dict(zip(SIZES, [False, False, True, False, True, True]))  # 100 은 합격이지만 150 이 불합격
    cells = make_cells({0.2: flags})
    assert minimum_size(cells, 0.2) == 200
    flags = dict(zip(SIZES, [False, True, True, True, True, True]))
    assert minimum_size(make_cells({0.2: flags}), 0.2) == 60


def test_minimum_size_none_when_largest_fails():
    cells = make_cells({0.2: dict(zip(SIZES, [True, True, True, True, True, False]))})
    assert minimum_size(cells, 0.2) is None


def test_recommend_is_worst_ratio_and_stops_when_missing():
    table = {0.2: dict(zip(SIZES, [False, False, False, True, True, True])),
             0.4: dict(zip(SIZES, [False, False, True, True, True, True])),
             0.6: dict(zip(SIZES, [False, True, True, True, True, True]))}
    rec = recommend(make_cells(table))
    assert rec["per_ratio"] == {0.2: 150, 0.4: 100, 0.6: 60} and rec["overall"] == 150
    table[0.2] = dict(zip(SIZES, [False] * 6))
    rec = recommend(make_cells(table))
    assert rec["per_ratio"][0.2] is None and rec["overall"] is None


# ---------- 한 회 실행·전체 실행 (학습은 가짜) ----------

def class_metrics(acc, recall=1.0):
    return ClassMetrics(accuracy=acc, recall=recall, precision=0.5, f1=0.5, confusion_matrix=[[1, 1], [0, 1]], n=3)


def model_metrics(version, drift_acc, drift_recall=1.0, normal_acc=1.0):
    return ModelMetrics(version=version, threshold=0.5, normal=class_metrics(normal_acc), drift=class_metrics(drift_acc, drift_recall),
                        train_data_snapshot={"train": 1, "pool": 0}, created_at=datetime.now(timezone.utc))


BASELINE = model_metrics("v1", 0.854)


def fake_data():
    split = SimpleNamespace(images=np.zeros((2, 128, 128), np.uint8), labels=["OK", "NG"], kinds=[None, None])
    return {"train": split, "test_normal": split, "test_drift": split}


def dark_dependent_fit(data, pool, seed):
    """dark 가 많을수록 G1 이 좋아지는 가짜 학습 (dark 8장 이상이면 +1%p 이상)."""
    return {"metrics": model_metrics("v2", 0.854 + 0.00125 * pool.n_dark), "kind_accuracy": {"dark": 0.8 + 0.01 * pool.n_dark, "noise": 0.6, "shift": 0.7}, "seconds": 0.5}


def test_run_one_computes_gates_and_records_dark_info(tmp_path):
    low = run_one(fake_data(), BASELINE, 6, 0.2, 0, dark_dependent_fit, tmp_path / "a", set())   # dark 1장
    high = run_one(fake_data(), BASELINE, 6, 0.6, 0, dark_dependent_fit, tmp_path / "b", set())  # dark 4장
    assert (low["n_normal"], low["n_dark"]) == (5, 1) and (high["n_normal"], high["n_dark"]) == (2, 4)
    assert not low["passed"] and not low["checks"]["G1"]["passed"]
    assert low["checks"]["G1"]["diff"] == pytest.approx(0.00125 * 1)
    assert low["checks"]["G2"]["passed"] and low["checks"]["G3"]["passed"]
    assert low["kind_accuracy"] == {"dark": pytest.approx(0.81), "noise": 0.6, "shift": 0.7} and isinstance(low["n_ng"], int)
    assert low["image_seed"] == 43 and low["train_seed"] == 42


def test_run_one_fails_when_pool_overlaps_dataset(tmp_path):
    pool = make_pool(3, 1, 43, tmp_path / "x")
    with pytest.raises(DatasetError):
        run_one(fake_data(), BASELINE, 4, 0.25, 0, dark_dependent_fit, tmp_path / "y", md5s(pool.images))


def make_root(tmp_path):
    root = tmp_path / "proj"
    (root / "models" / "v1").mkdir(parents=True)
    (root / "models" / "v1" / "model.pt").write_bytes(b"model")
    (root / "data").mkdir()
    (root / "data" / "mlops.db").write_bytes(b"db")
    (root / "data" / "pool").mkdir()
    (root / "data" / "pool" / "INS-000001.png").write_bytes(b"png")
    return root


def test_run_measure_report_and_hashes(tmp_path):
    root = make_root(tmp_path)
    logs = []
    out = run_measure(root, reps=2, sizes=(4, 5), fit_fn=dark_dependent_fit, data=fake_data(), baseline=BASELINE,
                      dataset_hashes=set(), log=logs.append)
    assert len(out["runs"]) == 2 * 3 * 2 and out["unchanged"]
    assert out["report_path"] == root / "data" / "pool_recovery_report.txt"
    text = out["report"]
    for needle in ("판정 규칙 (측정 전에 고정)", "선택 결과", "규칙 개정", "칸별 결과", "pool 안 정상", "dark 구간 정확도 평균", "noise 구간 정확도 평균 0.600", "shift 구간 정확도 평균 0.700", "shift 구간 정확도 0.700",
                   "통과율 ", "회차 1 (이미지 시드 43, 학습 시드 42)", "전체: 일치", "data/pool (1개)", "models/v1/model.pt"):
        assert needle in text
    assert "시연 권장 최소 건수: 없음" in text  # 크기 4·5 로는 G1 을 못 넘는다 (가짜 규칙)
    assert logs[0].startswith("측정 시작: 학습 12회") and (root / "data" / "pool_recovery_report.txt").read_text(encoding="utf-8").strip() == text


def test_run_measure_smoke_has_no_selection(tmp_path):
    root = make_root(tmp_path)
    out = run_measure(root, reps=5, smoke=True, sizes=(4,), fit_fn=dark_dependent_fit, data=fake_data(), baseline=BASELINE,
                      dataset_hashes=set(), log=lambda s: None)
    assert len(out["runs"]) == 3 and out["recommend"] is None  # 반복은 1회로 고정
    assert out["report_path"].name == "pool_recovery_smoke_report.txt"
    assert "스모크 실행이라 선택 규칙을 적용하지 않았다" in out["report"] and "판정 없음" in out["report"]


def test_run_measure_detects_changed_read_only_file(tmp_path):
    root = make_root(tmp_path)

    def tampering_fit(data, pool, seed):
        (root / "models" / "v1" / "model.pt").write_bytes(b"changed")
        return dark_dependent_fit(data, pool, seed)

    out = run_measure(root, reps=1, sizes=(4,), fit_fn=tampering_fit, data=fake_data(), baseline=BASELINE,
                      dataset_hashes=set(), log=lambda s: None)
    assert not out["unchanged"] and "전체: 불일치" in out["report"]


def test_fingerprint_covers_db_v1_and_pool(tmp_path):
    root = make_root(tmp_path)
    before = fingerprint(root)
    assert set(before) == {"data/mlops.db", "models/v1/model.pt", "data/pool (1개)"}
    (root / "data" / "pool" / "INS-000002.png").write_bytes(b"png2")
    after = fingerprint(root)
    assert "data/pool (2개)" in after and after["data/mlops.db"] == before["data/mlops.db"]
