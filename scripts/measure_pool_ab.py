"""재학습 안정성 측정 (측정만): 학습 시드와 이미지 시드(pool 구성) 중 무엇이 흔들림의 원인인지 따로 본다.

실행: python -m scripts.measure_pool_ab [--sizes 60,100,150] [--image-seeds 101,102,103,104] [--train-seeds 43,44,45,46]
      python -m scripts.measure_pool_ab --smoke     (기준점만. 직전 보고서와 같은 값인지 재현 확인, 판정 없음)

- 기준점: 이미지 시드 43 + 학습 시드 42 (실제 시연 재현). 크기마다 1회, 실험 A·B가 함께 쓴다.
- 실험 A: 학습 시드 42 고정, 이미지 시드만 바꾼다 ("pool 구성이 달라져도 유지되는가").
- 실험 B: 이미지 시드 43 고정, 학습 시드만 바꾼다 ("42가 운 좋은 시드인가").
- dark 비율은 60% 하나이고, 결론은 다른 비율로 넓히지 않는다.
- pool 만들기·학습·게이트·읽기 전용 파일 해시는 scripts/measure_pool_recovery.py 의 것을 그대로 쓴다 (그 파일은 수정하지 않는다).
- 서버·DB 는 쓰지 않는다. models/v1, data/mlops.db, data/pool 은 읽기만 한다 (시작·끝 해시 비교).
- 합격 규칙은 측정 전에 고정한 것이다 (RULE_LINES). 결과를 본 뒤 바꾸면 보고서의 '규칙 개정' 절에 사유와 함께 남긴다.
- 보고서에는 숫자와 합격/불합격만 쓰고 해석은 쓰지 않는다. 기존 data/pool_recovery_report.txt 는 덮어쓰지 않는다.
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from common.schemas import ModelMetrics
from scripts.make_dataset import PROJECT_ROOT, DatasetError, check_no_overlap, load_dataset
from scripts.measure_pool_recovery import (
    DEFAULT_DEFECT_RATE,
    KINDS,
    LABEL_SEED,
    dataset_md5s,
    demo_image_seed,
    fingerprint,
    make_pool,
    md5s,
    pool_plan,
    real_fit,
    train_seed_for,
)
from services.training.gate import evaluate_gate

REPORT_NAME = "pool_recovery_ab_report.txt"
SMOKE_REPORT_NAME = "pool_recovery_ab_smoke_report.txt"
OLD_REPORT_NAME = "pool_recovery_report.txt"   # 직전 90회 보고서 (덮어쓰지 않는다)
SIZES = (60, 100, 150)
RATIO = 0.6
A_IMAGE_SEEDS = (101, 102, 103, 104)
B_TRAIN_SEEDS = (43, 44, 45, 46)
STABLE_RATE = 0.8        # 안정: 기준점 포함 점 중 합격 비율 >= 80% (5개 중 4개)
TOLERANCE = 1e-9
BASE_G1_TOLERANCE_PP = 0.05   # 기준점 재현 확인에서 G1 차이 허용 오차 (%p)

# 직전 90회 보고서(data/pool_recovery_report.txt)의 같은 칸 (이미지 시드 43, 학습 시드 42, dark 60%): (합격 여부, G1 차이 %p)
EXPECTED_BASE = {60: (False, -5.40), 100: (True, 7.00), 150: (True, 14.60)}

CAUSE_TEXT = {
    "train_seed": "학습 시드가 주원인 (A 안정, B 불안정)",
    "image": "pool 구성(이미지)이 주원인 (A 불안정, B 안정)",
    "both": "둘 다 영향 (A·B 모두 불안정)",
    "hold": "판정 보류 (A·B 모두 안정, 직전 결과와 다르다)",
}

RULE_LINES = [
    "한 회 합격: G1·G2·G3 모두 통과",
    f"안정: 기준점을 포함한 점 중 합격 비율 >= {STABLE_RATE * 100:.0f}% (5개 중 4개 이상)",
    "원인 판정 (크기마다): A 안정 + B 불안정 = 학습 시드가 주원인, A 불안정 + B 안정 = pool 구성이 주원인, "
    "둘 다 불안정 = 둘 다 영향, 둘 다 안정 = 판정 보류하고 보고",
    "전체 요약: 판정별로 해당하는 크기의 수만 적는다 (다수결로 결론 내리지 않는다)",
    "기준점 재현 확인: 직전 보고서의 같은 칸과 합격 여부가 같고 G1 차이가 "
    f"{BASE_G1_TOLERANCE_PP}%p 이내여야 한다. 다르면 판정을 보류하고 멈춰서 보고한다",
    "G1만 실패 / G2 실패 / G3 실패 건수와 dark·noise·shift 구간 정확도는 기록만 하고 판정에 쓰지 않는다",
    "기준을 바꿔야 하면 멈춰서 보고하고, 결과를 본 뒤 바꾸면 '규칙 개정' 절에 사유와 함께 남긴다",
]

USE_LINES = [
    "시연은 학습 시드 42, 이미지 시드 43으로 항상 같은 조건이다. 기준점 외 시드의 결과는 그 조건이 얼마나 운에 달렸는지 보는 근거이다.",
    "A와 B가 모두 불안정이면 시연 조건은 우연히 통과할 수 있으므로, 대본은 리허설에서 사전 통과를 확인한 조합만 사용",
    "dark 60%에서만 측정했으므로 20%·40%에는 이 결론을 넓히지 않는다.",
]


# ---------- 실행 목록 ----------

def base_seeds() -> tuple:
    """(이미지 시드, 학습 시드) = 실제 시연 재현."""
    return demo_image_seed(), train_seed_for(0)


def build_plan(sizes=SIZES, image_seeds=A_IMAGE_SEEDS, train_seeds=B_TRAIN_SEEDS) -> list:
    """크기마다 기준점 1회 + 실험 A(이미지 시드 변경) + 실험 B(학습 시드 변경)."""
    base_image, base_train = base_seeds()
    plan = []
    for size in sizes:
        plan.append({"exp": "base", "size": size, "image_seed": base_image, "train_seed": base_train})
        plan += [{"exp": "A", "size": size, "image_seed": s, "train_seed": base_train} for s in image_seeds]
        plan += [{"exp": "B", "size": size, "image_seed": base_image, "train_seed": s} for s in train_seeds]
    return plan


def run_one(data, baseline: ModelMetrics, item: dict, fit_fn, work_dir, dataset_hashes) -> dict:
    n_normal, n_dark = pool_plan(item["size"], RATIO)
    pool = make_pool(n_normal, n_dark, item["image_seed"], work_dir)
    check_no_overlap({"pool": md5s(pool.images), "data/dataset": dataset_hashes})
    fit = fit_fn(data, pool, item["train_seed"])
    gate = evaluate_gate(baseline, fit["metrics"])
    checks = {c.id: {"passed": c.passed, "diff": c.candidate - c.baseline} for c in gate.checks}
    return {**item, "n_normal": n_normal, "n_dark": n_dark, "passed": gate.passed, "checks": checks,
            "kind_accuracy": dict(fit["kind_accuracy"]), "drift_accuracy": fit["metrics"].drift.accuracy,
            "seconds": fit["seconds"]}


# ---------- 판정 ----------

def stable(points: list) -> dict:
    passed = sum(r["passed"] for r in points)
    return {"passed": passed, "total": len(points), "stable": passed / len(points) >= STABLE_RATE - TOLERANCE}


def decide_cause(a_stable: bool, b_stable: bool) -> str:
    if a_stable and not b_stable:
        return "train_seed"
    if b_stable and not a_stable:
        return "image"
    return "hold" if a_stable and b_stable else "both"


def judge_size(runs: list, size: int) -> dict:
    mine = [r for r in runs if r["size"] == size]
    base = [r for r in mine if r["exp"] == "base"]
    a = stable(base + [r for r in mine if r["exp"] == "A"])
    b = stable(base + [r for r in mine if r["exp"] == "B"])
    return {"A": a, "B": b, "cause": decide_cause(a["stable"], b["stable"])}


def check_base(runs: list) -> list:
    """기준점이 직전 보고서와 같은가. [(크기, 일치 여부, 설명)]. 직전 보고서에 없는 크기는 건너뛴다."""
    out = []
    for r in runs:
        if r["exp"] != "base" or r["size"] not in EXPECTED_BASE:
            continue
        want_ok, want_g1 = EXPECTED_BASE[r["size"]]
        g1 = r["checks"]["G1"]["diff"] * 100
        ok = r["passed"] == want_ok and abs(g1 - want_g1) <= BASE_G1_TOLERANCE_PP
        out.append((r["size"], ok, f"이번 {'합격' if r['passed'] else '불합격'} G1 {g1:+.2f}%p / 직전 {'합격' if want_ok else '불합격'} G1 {want_g1:+.2f}%p"))
    return out


def failure_counts(runs: list) -> dict:
    fails = [r for r in runs if not r["passed"]]
    only_g1 = sum(not r["checks"]["G1"]["passed"] and r["checks"]["G2"]["passed"] and r["checks"]["G3"]["passed"] for r in fails)
    return {"runs": len(runs), "passed": len(runs) - len(fails), "only_g1": only_g1,
            **{g: sum(not r["checks"][g]["passed"] for r in runs) for g in ("G1", "G2", "G3")}}


# ---------- 측정 ----------

def run_measure(root=PROJECT_ROOT, smoke=False, sizes=SIZES, image_seeds=A_IMAGE_SEEDS, train_seeds=B_TRAIN_SEEDS, fit_fn=real_fit,
                data=None, baseline=None, dataset_hashes=None, report_path=None, log=print) -> dict:
    root = Path(root)
    plan = build_plan(sizes, image_seeds, train_seeds)
    if smoke:
        plan = [p for p in plan if p["exp"] == "base"]
    start_print = fingerprint(root)
    if data is None:
        data, _ = load_dataset(root)
    if baseline is None:
        baseline = ModelMetrics.from_dict(json.loads((root / "models" / "v1" / "metrics.json").read_text(encoding="utf-8")))
    if dataset_hashes is None:
        dataset_hashes = dataset_md5s(root)
    log(f"측정 시작: 학습 {len(plan)}회 (크기 {list(sizes)}, dark {int(RATIO * 100)}%{', 스모크: 기준점만' if smoke else ''})")
    runs, began = [], time.perf_counter()
    for item in plan:
        with tempfile.TemporaryDirectory() as work:
            runs.append(run_one(data, baseline, item, fit_fn, work, dataset_hashes))
        r = runs[-1]
        log(f"[{len(runs)}/{len(plan)}] 실험 {item['exp']} 크기 {item['size']} 이미지 시드 {item['image_seed']} 학습 시드 {item['train_seed']}: "
            f"{'합격' if r['passed'] else '불합격'} (경과 {time.perf_counter() - began:.0f}초)")
    base_check = check_base(runs)
    reproduced = all(ok for _, ok, _ in base_check)
    judged = None if smoke or not reproduced else {size: judge_size(runs, size) for size in sizes}
    end_print = fingerprint(root)
    report = build_report(runs, judged, base_check, reproduced, smoke, sizes, image_seeds, train_seeds, baseline, start_print, end_print)
    default_name = SMOKE_REPORT_NAME if smoke else REPORT_NAME
    report_path = Path(report_path) if report_path is not None else root / "data" / default_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")
    return {"runs": runs, "judged": judged, "base_check": base_check, "reproduced": reproduced, "report": report,
            "report_path": report_path, "unchanged": start_print == end_print}


# ---------- 보고서 (숫자와 합격/불합격만) ----------

def _ok(flag) -> str:
    return "합격" if flag else "불합격"


def _pp(x) -> str:
    return f"{x * 100:+.2f}%p"


def _run_line(r: dict) -> str:
    c = r["checks"]
    gates = ", ".join(f"{g} {_pp(c[g]['diff'])} {'통과' if c[g]['passed'] else '미통과'}" for g in ("G1", "G2", "G3"))
    kinds = ", ".join(f"{k} 구간 정확도 {r['kind_accuracy'][k]:.3f}" for k in KINDS)
    return (f"    실험 {r['exp']} (이미지 시드 {r['image_seed']}, 학습 시드 {r['train_seed']}): {gates}, "
            f"test_drift 정확도 {r['drift_accuracy']:.3f}, {kinds}, 학습 {r['seconds']:.1f}초 -> {_ok(r['passed'])}")


def build_report(runs, judged, base_check, reproduced, smoke, sizes, image_seeds, train_seeds, baseline, start_print, end_print) -> str:
    lines = ["== 학습 시드·이미지 시드 분리 측정" + (" (스모크: 기준점만, 판정 없음) ==" if smoke else " =="), "",
             "== 판정 규칙 (측정 전에 고정) =="] + RULE_LINES
    lines += ["", "== 기준점 재현 확인 (직전 90회 보고서의 같은 칸) =="]
    if not base_check:
        lines.append("비교할 칸이 없다 (직전 보고서에 없는 크기)")
    for size, ok, text in base_check:
        lines.append(f"크기 {size}: {'일치' if ok else '불일치'} ({text})")
    lines.append("재현 확인: " + ("일치" if reproduced else "불일치, 판정을 보류하고 멈춰서 보고한다"))
    lines += ["", "== 판정 결과 =="]
    if smoke:
        lines.append("스모크 실행이라 원인 판정을 하지 않았다")
    elif judged is None:
        lines.append("기준점 재현이 일치하지 않아 판정을 보류했다")
    else:
        for size, j in judged.items():
            lines.append(f"크기 {size}: A {j['A']['passed']}/{j['A']['total']} {'안정' if j['A']['stable'] else '불안정'}, "
                         f"B {j['B']['passed']}/{j['B']['total']} {'안정' if j['B']['stable'] else '불안정'} -> {CAUSE_TEXT[j['cause']]}")
        for cause, text in CAUSE_TEXT.items():
            count = sum(j["cause"] == cause for j in judged.values())
            lines.append(f"{text}: {count}개 크기")
    lines += ["", "== 규칙 개정 (결과 확인 후) ==", "없음", "", "== 결과 사용처 =="] + USE_LINES
    b = baseline
    image_base, train_base = base_seeds()
    lines += ["", "== 설정 ==",
              f"기준 모델(models/v1, 읽기 전용): test_drift 정확도 {b.drift.accuracy:.3f}, 재현율 {b.drift.recall:.3f}, test_normal 정확도 {b.normal.accuracy:.3f}",
              f"크기: {', '.join(map(str, sizes))}, dark 비율 {int(RATIO * 100)}%",
              f"기준점: 이미지 시드 {image_base}, 학습 시드 {train_base}",
              f"실험 A: 학습 시드 {train_base} 고정, 이미지 시드 {', '.join(map(str, image_seeds))}",
              f"실험 B: 이미지 시드 {image_base} 고정, 학습 시드 {', '.join(map(str, train_seeds))}",
              f"불량 순서: random.Random({LABEL_SEED}), 불량률 {DEFAULT_DEFECT_RATE}",
              f"총 학습 {len(runs)}회, 학습 시간 평균 {np.mean([r['seconds'] for r in runs]):.1f}초, 최대 {max(r['seconds'] for r in runs):.1f}초"]
    fc = failure_counts(runs)
    lines += ["", "== 실패 유형 (기록만, 판정에 쓰지 않음) ==",
              f"합격 {fc['passed']}/{fc['runs']}회, G1만 실패 {fc['only_g1']}회, G1 실패 {fc['G1']}회, G2 실패 {fc['G2']}회, G3 실패 {fc['G3']}회"]
    lines += ["", "== 회차별 결과 =="]
    for size in sizes:
        mine = [r for r in runs if r["size"] == size]
        first = mine[0]
        lines.append(f"크기 {size}: pool 안 정상 {first['n_normal']}장, dark {first['n_dark']}장")
        lines += [_run_line(r) for r in mine]
    lines += ["", "== 읽기 전용 파일 해시 (시작 -> 끝) =="]
    for name in sorted(set(start_print) | set(end_print)):
        a, e = start_print.get(name), end_print.get(name)
        lines.append(f"{name}: {'일치' if a == e else '불일치'} (sha256 {a})")
    lines.append(f"전체: {'일치' if start_print == end_print else '불일치'}")
    return "\n".join(lines)


def parse_ints(name: str, text: str) -> tuple:
    try:
        values = tuple(int(s) for s in text.split(","))
    except ValueError:
        raise ValueError(f"{name}: 60,100,150 처럼 쉼표로 나눈 정수여야 합니다 (받은 값: {text!r})") from None
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{name}: 비어 있거나 같은 값이 겹칩니다 (받은 값: {text!r})")
    return values


def main(argv=None, root=PROJECT_ROOT) -> int:
    parser = argparse.ArgumentParser(description="학습 시드와 이미지 시드를 따로 바꿔 재학습 안정성을 재고 data/pool_recovery_ab_report.txt 에 쓴다")
    parser.add_argument("--smoke", action="store_true", help="기준점만 (직전 보고서 재현 확인, 판정 없음, pool_recovery_ab_smoke_report.txt)")
    parser.add_argument("--sizes", default=",".join(map(str, SIZES)), help="pool 크기 (기본값 60,100,150)")
    parser.add_argument("--image-seeds", default=",".join(map(str, A_IMAGE_SEEDS)), help="실험 A 이미지 시드 (기본값 101,102,103,104)")
    parser.add_argument("--train-seeds", default=",".join(map(str, B_TRAIN_SEEDS)), help="실험 B 학습 시드 (기본값 43,44,45,46)")
    args = parser.parse_args(argv)
    try:
        sizes = parse_ints("--sizes", args.sizes)
        image_seeds = parse_ints("--image-seeds", args.image_seeds)
        train_seeds = parse_ints("--train-seeds", args.train_seeds)
        image_base, train_base = base_seeds()
        if image_base in image_seeds or train_base in train_seeds:
            raise ValueError(f"기준점 시드(이미지 {image_base}, 학습 {train_base})는 실험 시드에 넣을 수 없습니다 (같은 점이 겹칩니다)")
        out = run_measure(root, smoke=args.smoke, sizes=sizes, image_seeds=image_seeds, train_seeds=train_seeds)
    except (DatasetError, ValueError, OSError) as e:
        print(f"실패: {e}", file=sys.stderr)
        return 1
    print(f"보고서 저장: {out['report_path']}")
    print("기준점 재현: " + ("일치" if out["reproduced"] else "불일치"))
    print("읽기 전용 파일 해시: " + ("일치" if out["unchanged"] else "불일치"))
    return 0 if out["unchanged"] and out["reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
