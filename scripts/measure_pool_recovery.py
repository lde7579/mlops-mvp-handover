"""pool 건수별 재학습 회복도 측정 (측정만): 시연과 같은 구성의 pool(정상 N1장 + dark N2장)로 v2 를 임시 학습해
게이트 G1~G3 통과 여부를 칸(pool 크기 x dark 비율)별로 재고 data/pool_recovery_report.txt 에 쓴다.

실행: python -m scripts.measure_pool_recovery [--reps 5]
      python -m scripts.measure_pool_recovery --smoke     (크기 100, 세 비율, 1회. 본 측정 전 확인용, 판정 없음)

- 서버·DB 는 쓰지 않는다. models/v1/metrics.json 은 읽기만 하고(게이트 기준), v1 은 다시 학습하지 않는다.
- pool 은 시연과 같은 규칙으로 만든다: Collector(시드 = 설정 시드 + 시작 번호 1 = 43) + 불량 순서 random.Random(LABEL_SEED), 불량률은
  simulator_runner 의 값 (backend.generation / backend.simulator_runner 와 같은 경로. 시험이 해시로 같음을 확인한다). 파일은 임시 폴더에만 쓴다.
- v2 는 서비스(backend/retrain.py)처럼 고정 train + pool 순서로 parts.predictor.train() 에 넣고, 평가는 고정 test_normal·test_drift 이다.
  테스트셋은 학습에 쓰지 않는다 (pool 이 data/dataset 과 겹치면 실패).
- 반복 k=0 은 실제 시연 재현(이미지 시드 43, 학습 시드 42), k>=1 은 이미지 시드 100+k, 학습 시드 42+k.
- 합격 규칙은 측정 전에 고정한 것이다 (RULE_LINES). 결과를 본 뒤 바꾸면 보고서의 '규칙 개정' 절에 사유와 함께 남긴다.
- 보고서에는 숫자와 합격/불합격만 쓰고 해석은 쓰지 않는다. docs/, common/, parts/ 는 수정하지 않는다.
"""
import argparse
import hashlib
import json
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.simulator_runner import DEFAULT_DEFECT_RATE, LABEL_SEED
from common.schemas import ModelMetrics
from parts.collector.collector import Collector
from parts.collector.config import load_config as load_collector_config
from parts.predictor.config import load_config as load_predictor_config
from scripts.make_dataset import PROJECT_ROOT, DatasetError, check_no_overlap, load_dataset
from services.training.gate import evaluate_gate

REPORT_NAME = "pool_recovery_report.txt"
SMOKE_REPORT_NAME = "pool_recovery_smoke_report.txt"
SIZES = (30, 60, 100, 150, 200, 300)
RATIOS = (0.2, 0.4, 0.6)
SMOKE_SIZES = (100,)
REPS = 5
PASS_RATE = 0.8          # 칸 합격: 통과율 >= 80% 이고 반복 0(실제 시연 재현)도 통과 (승인된 안)
ALT_IMAGE_SEED_BASE = 100
TOLERANCE = 1e-9
DARK = "dark"
KINDS = ("dark", "noise", "shift")
INTERVAL_SECONDS = 2

RULE_LINES = [
    "한 회 합격: G1·G2·G3 모두 통과",
    f"칸(pool 크기 x dark 비율) 합격: 반복 중 합격한 비율 >= {PASS_RATE * 100:.0f}% 이고 반복 1회차(실제 시연 재현)도 합격",
    "비율별 최소 pool 건수: 그 크기와 그보다 큰 모든 크기의 칸이 합격인 가장 작은 크기",
    "시연 권장 최소 건수: 비율별 최소 pool 건수 중 가장 큰 값",
    "합격 크기가 없는 비율이 있으면 기준을 바꾸지 않고 멈춰서 보고한다",
    "dark·noise·shift 구간 정확도(test_drift 중 그 종류 이미지만)와 pool 안 dark 장수는 기록만 하고 판정에 쓰지 않는다",
]


@dataclass
class Pool:
    images: np.ndarray   # uint8 (N, 128, 128), 정상 n_normal 장 뒤에 dark n_dark 장
    labels: list
    n_normal: int
    n_dark: int


# ---------- pool 만들기 (시연과 같은 규칙) ----------

def pool_plan(size: int, ratio: float) -> tuple:
    """(정상 장수, dark 장수). dark 장수 = 반올림(크기 x 비율)."""
    n_dark = round(size * ratio)
    return size - n_dark, n_dark


def demo_image_seed() -> int:
    """실제 시연의 수집기 시드: 설정 시드 + 시작 번호 1 (backend/generation.py 의 _ready 와 같은 규칙)."""
    return load_collector_config().seed + 1


def image_seed_for(rep: int) -> int:
    return demo_image_seed() if rep == 0 else ALT_IMAGE_SEED_BASE + rep


def train_seed_for(rep: int) -> int:
    return load_predictor_config().seed + rep


def make_pool(n_normal: int, n_dark: int, image_seed: int, work_dir) -> Pool:
    """정상 n_normal 장 뒤에 dark n_dark 장. 불량 순서와 시나리오는 simulator_runner._cycle 과 같다 (파일은 work_dir 에만 쓴다)."""
    collector = Collector(root=work_dir, seed=image_seed)
    rng = random.Random(LABEL_SEED)
    images, labels = [], []
    for i in range(n_normal + n_dark):
        label = "NG" if rng.random() < DEFAULT_DEFECT_RATE else "OK"
        drift = DARK if i >= n_normal else None
        sample = collector.next_sample(scenario="drift" if drift else "normal", data_split="pool", label=label, drift_type=drift)
        images.append(sample.image)
        labels.append(label)
    return Pool(np.stack(images), labels, n_normal, n_dark)


def md5s(images) -> set:
    return {hashlib.md5(np.ascontiguousarray(im).tobytes()).hexdigest() for im in images}


def dataset_md5s(root) -> set:
    lines = (Path(root) / "data" / "dataset" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    return {json.loads(line)["image_md5"] for line in lines}


# ---------- 학습·게이트 ----------

def real_fit(data, pool: Pool, seed: int) -> dict:
    """고정 train + pool 로 v2 를 임시 폴더에 학습하고, 게이트에 쓸 지표와 test_drift dark 정확도를 돌려준다."""
    from parts.predictor import Predictor, evaluate, train  # 여기서 처음 torch 를 가져온다

    images = np.concatenate([data["train"].images, pool.images])
    labels = list(data["train"].labels) + list(pool.labels)
    snapshot = {"train": len(data["train"].labels), "pool": len(pool.labels)}
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        metrics = train(images, labels, version="v2",
                        eval_normal=(data["test_normal"].images, data["test_normal"].labels),
                        eval_drift=(data["test_drift"].images, data["test_drift"].labels),
                        snapshot=snapshot, seed=seed, root=tmp)
        seconds = time.perf_counter() - start
        predictor = Predictor(root=tmp)
        predictor.load("v2")
        kind_accuracy = {}
        for kind in KINDS:  # test_drift 중 그 종류 이미지만의 정확도 (참고 기록, 판정에 쓰지 않는다)
            mask = np.array([k == kind for k in data["test_drift"].kinds])
            kind_accuracy[kind] = evaluate(predictor, data["test_drift"].images[mask],
                                           [lab for lab, m in zip(data["test_drift"].labels, mask) if m], predictor.threshold).accuracy
    return {"metrics": metrics, "kind_accuracy": kind_accuracy, "seconds": seconds}


def run_one(data, baseline: ModelMetrics, size: int, ratio: float, rep: int, fit_fn, work_dir, dataset_hashes) -> dict:
    n_normal, n_dark = pool_plan(size, ratio)
    image_seed, train_seed = image_seed_for(rep), train_seed_for(rep)
    pool = make_pool(n_normal, n_dark, image_seed, work_dir)
    check_no_overlap({"pool": md5s(pool.images), "data/dataset": dataset_hashes})
    fit = fit_fn(data, pool, train_seed)
    gate = evaluate_gate(baseline, fit["metrics"])
    checks = {c.id: {"passed": c.passed, "diff": c.candidate - c.baseline, "candidate": c.candidate} for c in gate.checks}
    return {"size": size, "ratio": ratio, "rep": rep, "image_seed": image_seed, "train_seed": train_seed,
            "n_normal": n_normal, "n_dark": n_dark, "n_ng": sum(lab == "NG" for lab in pool.labels),
            "passed": gate.passed, "checks": checks, "kind_accuracy": dict(fit["kind_accuracy"]),
            "drift_accuracy": fit["metrics"].drift.accuracy, "seconds": fit["seconds"]}


# ---------- 집계·선택 규칙 ----------

def cell_summary(runs: list) -> dict:
    """한 칸의 반복 결과 목록 -> 통과율, 반복 0 통과 여부, 칸 합격."""
    ordered = sorted(runs, key=lambda r: r["rep"])
    passed = sum(r["passed"] for r in ordered)
    rate = passed / len(ordered)
    first = next((r for r in ordered if r["rep"] == 0), None)
    first_ok = bool(first and first["passed"])
    return {"runs": ordered, "passed": passed, "total": len(ordered), "rate": rate, "first_ok": first_ok,
            "ok": rate >= PASS_RATE - TOLERANCE and first_ok}


def group_cells(runs: list) -> dict:
    grouped = {}
    for r in runs:
        grouped.setdefault((r["size"], r["ratio"]), []).append(r)
    return {key: cell_summary(v) for key, v in grouped.items()}


def minimum_size(cells: dict, ratio: float, sizes=SIZES):
    """그 비율에서 '그 크기와 더 큰 모든 크기'가 칸 합격인 가장 작은 크기. 없으면 None."""
    found = None
    for size in sorted(sizes, reverse=True):
        if not cells[(size, ratio)]["ok"]:
            break
        found = size
    return found


def recommend(cells: dict, ratios=RATIOS, sizes=SIZES) -> dict:
    per_ratio = {r: minimum_size(cells, r, sizes) for r in ratios}
    overall = None if any(v is None for v in per_ratio.values()) else max(per_ratio.values())
    return {"per_ratio": per_ratio, "overall": overall}


# ---------- 읽기 전용 파일 해시 ----------

def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(root) -> dict:
    """data/mlops.db(-wal, -shm), models/v1/*, data/pool 전체의 sha256. 없는 파일은 건너뛴다. pool 은 파일 수와 통합 해시."""
    root = Path(root)
    paths = [root / "data" / name for name in ("mlops.db", "mlops.db-wal", "mlops.db-shm")]
    v1 = root / "models" / "v1"
    if v1.is_dir():
        paths += sorted(p for p in v1.iterdir() if p.is_file())
    out = {p.relative_to(root).as_posix(): _sha(p) for p in paths if p.is_file()}
    pool = root / "data" / "pool"
    files = sorted(p for p in pool.rglob("*") if p.is_file()) if pool.is_dir() else []
    joined = "\n".join(f"{p.relative_to(pool).as_posix()}:{_sha(p)}" for p in files)
    out[f"data/pool ({len(files)}개)"] = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return out


# ---------- 측정 ----------

def run_measure(root=PROJECT_ROOT, reps=REPS, smoke=False, sizes=None, ratios=RATIOS, fit_fn=real_fit, data=None,
                baseline=None, dataset_hashes=None, report_path=None, log=print) -> dict:
    root = Path(root)
    sizes = tuple(sizes) if sizes is not None else (SMOKE_SIZES if smoke else SIZES)
    reps = 1 if smoke else reps
    start_print = fingerprint(root)
    if data is None:
        data, _ = load_dataset(root)
    if baseline is None:
        baseline = ModelMetrics.from_dict(json.loads((root / "models" / "v1" / "metrics.json").read_text(encoding="utf-8")))
    if dataset_hashes is None:
        dataset_hashes = dataset_md5s(root)
    total = len(sizes) * len(ratios) * reps
    log(f"측정 시작: 학습 {total}회 (크기 {list(sizes)}, dark 비율 {list(ratios)}, 반복 {reps}회)")
    runs, began = [], time.perf_counter()
    for size in sizes:
        for ratio in ratios:
            for rep in range(reps):
                with tempfile.TemporaryDirectory() as work:
                    runs.append(run_one(data, baseline, size, ratio, rep, fit_fn, work, dataset_hashes))
                r = runs[-1]
                log(f"[{len(runs)}/{total}] 크기 {size} dark {int(ratio * 100)}% 회차 {rep + 1}: "
                    f"{'합격' if r['passed'] else '불합격'} (경과 {time.perf_counter() - began:.0f}초)")
    cells = group_cells(runs)
    result = recommend(cells, ratios, sizes) if not smoke else None
    end_print = fingerprint(root)
    report = build_report(runs, cells, result, baseline, smoke, reps, sizes, ratios, start_print, end_print)
    default_name = SMOKE_REPORT_NAME if smoke else REPORT_NAME
    report_path = Path(report_path) if report_path is not None else root / "data" / default_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")
    return {"runs": runs, "cells": cells, "recommend": result, "report": report, "report_path": report_path,
            "unchanged": start_print == end_print}


# ---------- 보고서 (숫자와 합격/불합격만) ----------

def _ok(flag) -> str:
    return "합격" if flag else "불합격"


def _pp(x) -> str:
    return f"{x * 100:+.2f}%p"


def _kinds(r: dict) -> str:
    return ", ".join(f"{k} 구간 정확도 {r['kind_accuracy'][k]:.3f}" for k in KINDS)


def _run_line(r: dict) -> str:
    c = r["checks"]
    gates = ", ".join(f"{g} {_pp(c[g]['diff'])} {'통과' if c[g]['passed'] else '미통과'}" for g in ("G1", "G2", "G3"))
    return (f"    회차 {r['rep'] + 1} (이미지 시드 {r['image_seed']}, 학습 시드 {r['train_seed']}): {gates}, "
            f"test_drift 정확도 {r['drift_accuracy']:.3f}, {_kinds(r)}, 학습 {r['seconds']:.1f}초 -> {_ok(r['passed'])}")


def build_report(runs, cells, result, baseline, smoke, reps, sizes, ratios, start_print, end_print) -> str:
    title = "== pool 건수별 회복도 측정 (스모크: 판정 없음) ==" if smoke else "== pool 건수별 회복도 측정 =="
    lines = [title, "", "== 판정 규칙 (측정 전에 고정) =="] + RULE_LINES
    lines += ["", "== 선택 결과 =="]
    if smoke:
        lines.append("스모크 실행이라 선택 규칙을 적용하지 않았다")
    else:
        for ratio in ratios:
            size = result["per_ratio"][ratio]
            lines.append(f"dark {int(ratio * 100)}%: 최소 pool 건수 " + ("없음 (합격 크기 없음)" if size is None else f"{size}장"))
        overall = result["overall"]
        if overall is None:
            lines.append("시연 권장 최소 건수: 없음 (합격 크기가 없는 비율이 있다. 기준을 바꾸지 않고 멈춰서 보고한다)")
        else:
            lines.append(f"시연 권장 최소 건수: {overall}장 (시뮬레이터 {INTERVAL_SECONDS}초 간격으로 약 {overall * INTERVAL_SECONDS}초)")
    lines += ["", "== 규칙 개정 (결과 확인 후) ==", "없음"]
    b = baseline
    lines += ["", "== 설정 ==",
              f"기준 모델(models/v1, 읽기 전용): test_drift 정확도 {b.drift.accuracy:.3f}, 재현율 {b.drift.recall:.3f}, test_normal 정확도 {b.normal.accuracy:.3f}",
              f"pool 크기: {', '.join(map(str, sizes))}",
              f"dark 비율: {', '.join(f'{int(r * 100)}%' for r in ratios)}",
              f"반복: {reps}회 (1회차 = 실제 시연 재현: 이미지 시드 {image_seed_for(0)}, 학습 시드 {train_seed_for(0)})",
              f"불량 순서: random.Random({LABEL_SEED}), 불량률 {DEFAULT_DEFECT_RATE}",
              f"총 학습 {len(runs)}회, 학습 시간 평균 {np.mean([r['seconds'] for r in runs]):.1f}초, 최대 {max(r['seconds'] for r in runs):.1f}초"]
    lines += ["", "== 칸별 결과 =="]
    for size in sizes:
        for ratio in ratios:
            cell = cells[(size, ratio)]
            first = cell["runs"][0]
            means = ", ".join(f"{k} 구간 정확도 평균 {np.mean([r['kind_accuracy'][k] for r in cell['runs']]):.3f}" for k in KINDS)
            lines.append(f"크기 {size} / dark {int(ratio * 100)}%: pool 안 정상 {first['n_normal']}장, dark {first['n_dark']}장, 불량 {first['n_ng']}장")
            verdict = "판정 없음" if smoke else f"칸 {_ok(cell['ok'])}"
            lines.append(f"  통과율 {cell['passed']}/{cell['total']} ({cell['rate'] * 100:.1f}%), 1회차 {_ok(cell['first_ok'])}, "
                         f"{means}, {verdict}")
            lines += [_run_line(r) for r in cell["runs"]]
    lines += ["", "== 읽기 전용 파일 해시 (시작 -> 끝) =="]
    for name in sorted(set(start_print) | set(end_print)):
        a, e = start_print.get(name), end_print.get(name)
        lines.append(f"{name}: {'일치' if a == e else '불일치'} (sha256 {a})")
    lines.append(f"전체: {'일치' if start_print == end_print else '불일치'}")
    return "\n".join(lines)


def main(argv=None, root=PROJECT_ROOT) -> int:
    parser = argparse.ArgumentParser(description="pool 크기·dark 비율별 v2 게이트 통과율을 오프라인으로 재서 data/pool_recovery_report.txt 에 쓴다")
    parser.add_argument("--reps", type=int, default=REPS, help=f"칸마다 반복 횟수 (기본값 {REPS})")
    parser.add_argument("--smoke", action="store_true", help="크기 100, 세 비율, 1회만 (판정 없음, 보고서는 pool_recovery_smoke_report.txt)")
    args = parser.parse_args(argv)
    try:
        if args.reps < 1:
            raise ValueError(f"--reps: 1 이상의 정수여야 합니다 (받은 값: {args.reps})")
        out = run_measure(root, reps=args.reps, smoke=args.smoke)
    except (DatasetError, ValueError, OSError) as e:
        print(f"실패: {e}", file=sys.stderr)
        return 1
    print(f"보고서 저장: {out['report_path']}")
    if out["recommend"] is not None:
        overall = out["recommend"]["overall"]
        print("시연 권장 최소 건수: " + ("없음" if overall is None else f"{overall}장"))
    print("읽기 전용 파일 해시: " + ("일치" if out["unchanged"] else "불일치"))
    return 0 if out["unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
