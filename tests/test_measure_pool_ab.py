"""scripts/measure_pool_ab.py 테스트: 학습은 가짜로 바꾸고(torch 불필요), pool 은 아주 작게(크기 4) 만든다."""
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.measure_pool_ab as ab
from common.schemas import ClassMetrics, ModelMetrics
from scripts.measure_pool_ab import (
    A_IMAGE_SEEDS,
    B_TRAIN_SEEDS,
    REPORT_NAME,
    RULE_LINES,
    SMOKE_REPORT_NAME,
    USE_LINES,
    base_seeds,
    build_plan,
    check_base,
    decide_cause,
    failure_counts,
    judge_size,
    main,
    parse_ints,
    run_measure,
    stable,
)


# ---------- 실행 목록 ----------

def test_default_plan_matches_the_design():
    plan = build_plan()
    assert len(plan) == 27 and base_seeds() == (43, 42)
    for size in (60, 100, 150):
        mine = [p for p in plan if p["size"] == size]
        assert [p["exp"] for p in mine] == ["base"] + ["A"] * 4 + ["B"] * 4
        assert (mine[0]["image_seed"], mine[0]["train_seed"]) == (43, 42)
        assert [(p["image_seed"], p["train_seed"]) for p in mine[1:5]] == [(s, 42) for s in (101, 102, 103, 104)]  # A: 학습 시드 고정
        assert [(p["image_seed"], p["train_seed"]) for p in mine[5:]] == [(43, s) for s in (43, 44, 45, 46)]       # B: 이미지 시드 고정
    assert A_IMAGE_SEEDS == (101, 102, 103, 104) and B_TRAIN_SEEDS == (43, 44, 45, 46)


# ---------- 판정 ----------

def pts(*flags):
    return [{"passed": f} for f in flags]


def test_stable_boundary_is_four_of_five():
    assert stable(pts(True, True, True, True, False))["stable"]
    assert not stable(pts(True, True, True, False, False))["stable"]
    assert stable(pts(True, True, True, True, True)) == {"passed": 5, "total": 5, "stable": True}


def test_decide_cause_four_cases():
    assert decide_cause(True, False) == "train_seed"   # 이미지를 바꿔도 안정, 학습 시드를 바꾸면 불안정
    assert decide_cause(False, True) == "image"
    assert decide_cause(False, False) == "both"
    assert decide_cause(True, True) == "hold"


def run(exp, size, passed, g=(True, True, True), image_seed=0, train_seed=0):
    checks = {f"G{i + 1}": {"passed": ok, "diff": 0.01 if ok else -0.05} for i, ok in enumerate(g)}
    return {"exp": exp, "size": size, "passed": passed, "checks": checks, "image_seed": image_seed, "train_seed": train_seed}


def test_judge_size_uses_base_in_both_experiments():
    runs = [run("base", 60, True)] + [run("A", 60, True)] * 3 + [run("A", 60, False)] + [run("B", 60, True)] + [run("B", 60, False)] * 3
    j = judge_size(runs, 60)
    assert (j["A"]["passed"], j["A"]["total"], j["A"]["stable"]) == (4, 5, True)   # 기준 1 + A 3
    assert (j["B"]["passed"], j["B"]["total"], j["B"]["stable"]) == (2, 5, False)  # 기준 1 + B 1
    assert j["cause"] == "train_seed"
    other = [run("base", 100, False)] + [run("A", 100, False)] * 4 + [run("B", 100, True)] * 4
    assert judge_size(other + runs, 100)["cause"] == "image"  # 크기가 다르면 섞이지 않는다


def test_failure_counts():
    runs = [run("base", 60, True), run("A", 60, False, (False, True, True)), run("A", 60, False, (False, False, True)),
            run("B", 60, False, (True, False, False))]
    assert failure_counts(runs) == {"runs": 4, "passed": 1, "only_g1": 1, "G1": 2, "G2": 2, "G3": 1}


# ---------- 기준점 재현 확인 ----------

def base_run(size, passed, g1):
    r = run("base", size, passed)
    r["checks"]["G1"]["diff"] = g1
    return r


def test_check_base_match_and_mismatch():
    same = [base_run(60, False, -0.054), base_run(100, True, 0.07), base_run(150, True, 0.146)]
    assert all(ok for _, ok, _ in check_base(same))
    assert not dict((s, ok) for s, ok, _ in check_base([base_run(100, True, 0.08)]))[100]       # G1 차이가 0.05%p 넘게 다름
    assert not dict((s, ok) for s, ok, _ in check_base([base_run(100, False, 0.07)]))[100]      # 합격 여부가 다름
    assert check_base([base_run(4, True, 0.5), run("A", 100, True)]) == []                      # 직전 보고서에 없는 크기·실험은 건너뜀


# ---------- 전체 실행 (학습은 가짜) ----------

def class_metrics(acc, recall=1.0):
    return ClassMetrics(accuracy=acc, recall=recall, precision=0.5, f1=0.5, confusion_matrix=[[1, 1], [0, 1]], n=3)


def model_metrics(version, drift_acc):
    return ModelMetrics(version=version, threshold=0.5, normal=class_metrics(1.0), drift=class_metrics(drift_acc),
                        train_data_snapshot={"train": 1, "pool": 0}, created_at=datetime.now(timezone.utc))


BASELINE = model_metrics("v1", 0.854)
FAIL_TRAIN_SEEDS = {43, 44}


def fake_fit(data, pool, seed):
    """학습 시드가 FAIL_TRAIN_SEEDS 에 있으면 G1 미통과."""
    acc = 0.80 if seed in FAIL_TRAIN_SEEDS else 0.90
    return {"metrics": model_metrics("v2", acc), "kind_accuracy": {"dark": 0.9, "noise": 0.8, "shift": 0.7}, "seconds": 0.5}


def fake_data():
    split = SimpleNamespace(images=np.zeros((2, 128, 128), np.uint8), labels=["OK", "NG"], kinds=[None, None])
    return {"train": split, "test_normal": split, "test_drift": split}


def make_root(tmp_path):
    root = tmp_path / "proj"
    (root / "models" / "v1").mkdir(parents=True)
    (root / "models" / "v1" / "model.pt").write_bytes(b"model")
    (root / "data" / "pool").mkdir(parents=True)
    (root / "data" / "mlops.db").write_bytes(b"db")
    (root / "data" / "pool" / "INS-000001.png").write_bytes(b"png")
    return root


def measure(root, **kw):
    args = dict(sizes=(4,), image_seeds=(101, 102), train_seeds=(43, 46), fit_fn=fake_fit, data=fake_data(), baseline=BASELINE,
                dataset_hashes=set(), log=lambda s: None)
    args.update(kw)
    return run_measure(root, **args)


def test_run_measure_judges_and_reports(tmp_path):
    root = make_root(tmp_path)
    old = root / "data" / "pool_recovery_report.txt"
    old.write_text("직전 보고서", encoding="utf-8")
    logs = []
    out = measure(root, log=logs.append)
    assert len(out["runs"]) == 5 and out["unchanged"] and out["reproduced"]
    j = out["judged"][4]
    assert (j["A"]["passed"], j["A"]["total"]) == (3, 3) and (j["B"]["passed"], j["B"]["total"]) == (2, 3)  # B: 학습 시드 43 미통과
    assert logs[0].startswith("측정 시작: 학습 5회") and logs[-1].startswith("[5/5]")
    assert out["report_path"] == root / "data" / REPORT_NAME and REPORT_NAME != "pool_recovery_report.txt"
    assert old.read_text(encoding="utf-8") == "직전 보고서"  # 직전 보고서는 덮어쓰지 않는다
    text = out["report"]
    for needle in ("판정 규칙 (측정 전에 고정)", "기준점 재현 확인", "판정 결과", "규칙 개정", "결과 사용처", "회차별 결과", "실패 유형",
                   "실험 A (이미지 시드 101, 학습 시드 42)", "실험 B (이미지 시드 43, 학습 시드 46)", "noise 구간 정확도 0.800",
                   "전체: 일치", "data/pool (1개)", "models/v1/model.pt"):
        assert needle in text
    assert root.joinpath("data", REPORT_NAME).read_text(encoding="utf-8").strip() == text


def test_report_has_required_use_lines_and_rules():
    assert any("A와 B가 모두 불안정이면 시연 조건은 우연히 통과할 수 있으므로, 대본은 리허설에서 사전 통과를 확인한 조합만 사용" in line
               for line in USE_LINES)
    assert any("4개 이상" in line for line in RULE_LINES)


def test_report_use_section_contains_phrase(tmp_path):
    out = measure(make_root(tmp_path))
    section = out["report"].split("== 결과 사용처 ==")[1].split("== 설정 ==")[0]
    assert "A와 B가 모두 불안정이면 시연 조건은 우연히 통과할 수 있으므로, 대본은 리허설에서 사전 통과를 확인한 조합만 사용" in section


def test_mismatched_base_holds_judgement(tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "EXPECTED_BASE", {4: (False, 99.0)})
    out = measure(make_root(tmp_path))
    assert not out["reproduced"] and out["judged"] is None
    assert "재현 확인: 불일치" in out["report"] and "판정을 보류" in out["report"]


def test_smoke_runs_only_base(tmp_path):
    out = measure(make_root(tmp_path), smoke=True, sizes=(4, 5))
    assert [r["exp"] for r in out["runs"]] == ["base", "base"] and out["judged"] is None
    assert out["report_path"].name == SMOKE_REPORT_NAME
    assert "스모크 실행이라 원인 판정을 하지 않았다" in out["report"]


def test_changed_read_only_file_is_detected(tmp_path):
    root = make_root(tmp_path)

    def tampering(data, pool, seed):
        (root / "models" / "v1" / "model.pt").write_bytes(b"changed")
        return fake_fit(data, pool, seed)

    out = measure(root, fit_fn=tampering, smoke=True)
    assert not out["unchanged"] and "전체: 불일치" in out["report"]


# ---------- 명령줄 ----------

def test_parse_ints():
    assert parse_ints("--sizes", "60,100") == (60, 100)
    for bad in ("", "a,b", "1,1"):
        with pytest.raises(ValueError):
            parse_ints("--sizes", bad)


def test_main_rejects_base_seeds_in_experiments(tmp_path, capsys):
    assert main(["--image-seeds", "43,101"], root=tmp_path) == 1
    assert "같은 점이 겹칩니다" in capsys.readouterr().err
    assert main(["--train-seeds", "42,43"], root=tmp_path) == 1
    assert main(["--sizes", "x"], root=tmp_path) == 1
