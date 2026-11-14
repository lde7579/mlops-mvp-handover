"""드리프트 강도 스윕 (측정만): 종류별·강도별로 v1 이 얼마나 떨어지는지 data/sweep_report.txt 에 쓴다.

실행: python -m scripts.sweep_drift [--seeds 42,44] [--dark 0.05,0.1] [--noise ...] [--shift ...]

- v1 은 data/dataset/train 으로 학습한다 (3-B 의 v1 과 같은 데이터·설정·시드). 모델은 임시 폴더에만 저장한다.
- 평가 이미지는 종류·강도마다 정상 150장 + 불량 40장을 make_image() 로 직접 만든다 (고정 시드).
  학습·테스트 세트와 이미지가 하나도 겹치지 않아야 하고, 겹치면 실패한다. test_drift 는 강도를 고르는 데 쓰지 않는다.
- collector.toml, parts/, docs/ 는 수정하지 않는다. 강도를 바꿀지는 결과를 보고 사람이 정한다.
- 보고서에는 숫자와 목표 구간 안/밖 표시만 쓰고 해석은 쓰지 않는다.
"""
import argparse
import hashlib
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.schemas import validate_image_array
from parts.collector.simulator import BASE_NOISE, BOLT_COUNT, DRIFT_TYPES, make_image
from parts.predictor import Predictor, evaluate, train
from parts.predictor.config import load_config as load_predictor_config
from scripts.make_dataset import DATASET_DIR, MANIFEST, PROJECT_ROOT, DatasetError, check_no_overlap, load_dataset
from scripts.measure_baseline import parse_seeds

REPORT_NAME = "sweep_report.txt"
SWEEP_SEED = 9001  # 평가 이미지 시드 영역. 데이터셋 시드(1001~1004)와 다른 영역이다
N_NORMAL = 150
N_DEFECT = 40
TARGET = (0.70, 0.90)  # 목표 구간: 종류별 정확도 (경계 포함)
TOLERANCE = 1e-9
BASELINE = "정상 환경"
KIND_INDEX = {"dark": 0, "noise": 1, "shift": 2, BASELINE: 3}

# 강도는 낮은 쪽부터 5개 안팎. dark 는 현재값(0.30)에서 이미 v1 정확도가 목표보다 낮아 낮은 쪽으로 잡았다.
DEFAULT_STRENGTHS = {
    "dark": (0.05, 0.10, 0.15, 0.20, 0.30),
    "noise": (0.055, 0.11, 0.22, 0.33, 0.44),
    "shift": (0.02, 0.04, 0.08, 0.12, 0.16),
}


def parse_strengths(text: str, name: str = "강도") -> tuple:
    """'0.1,0.2' -> (0.1, 0.2). 0 초과 1 이하, 겹침 없음."""
    try:
        values = tuple(float(s) for s in text.split(","))
    except ValueError:
        raise ValueError(f"{name}: 0.1,0.2 처럼 쉼표로 나눈 숫자여야 합니다 (받은 값: {text!r})") from None
    if not values or len(set(values)) != len(values) or any(not 0 < v <= 1 for v in values):
        raise ValueError(f"{name}: 0 초과 1 이하의 서로 다른 숫자여야 합니다 (받은 값: {text!r})")
    return values


def describe(kind: str, s: float) -> str:
    """강도 s 가 실제로 뜻하는 값 (simulator.make_image 의 공식)."""
    if kind == "dark":
        return f"밝기 배율 {1 - 0.9 * s:.3f}"
    if kind == "noise":
        return f"노이즈 표준편차 {BASE_NOISE + 100 * s:.1f}"
    if kind == "shift":
        return f"위치 흔들림 ±{2 + 25 * s:.2f}픽셀"
    return "드리프트 없음"


def in_target(accuracy: float) -> bool:
    return TARGET[0] - TOLERANCE <= accuracy <= TARGET[1] + TOLERANCE


def false_alarm_rate(confusion_matrix) -> float:
    """오경보율 = 정상을 불량으로 판정한 비율 = FP / (FP + TN)."""
    (tn, fp), _ = confusion_matrix
    return fp / (fp + tn) if fp + tn else 0.0


# ---------- 평가 이미지 ----------

@dataclass
class Cell:
    images: np.ndarray  # uint8 (N, 128, 128)
    labels: list        # "OK" / "NG"
    unfastened: list    # 이미지마다 미체결 볼트 번호 (정상이면 [])


def make_cell(kind, strength: float, n_normal: int = N_NORMAL, n_defect: int = N_DEFECT) -> Cell:
    """종류·강도 한 칸의 평가 이미지. kind 가 BASELINE 이면 드리프트 없는 정상 환경.

    같은 (종류, 강도)는 목록 순서와 상관없이 항상 같은 이미지가 나온다 (강도 값으로 시드를 만든다).
    """
    drift = None if kind == BASELINE else kind
    rng = np.random.default_rng(np.random.SeedSequence([SWEEP_SEED, KIND_INDEX[kind], int(round(strength * 10000))]))
    labels = ["NG"] * n_defect + ["OK"] * n_normal
    labels = [labels[i] for i in rng.permutation(len(labels))]
    images, parts = [], []
    for label in labels:
        unfastened = []
        if label == "NG":
            # 불량이면 볼트 6개 중 1~2개를 무작위로 미체결(번호 1~6)로 한다.
            # 근거: parts/collector/collector.py 의 next_sample() 과 같은 규칙 (계약 2장 missing_parts 와
            # 스파이크의 "불량이면 1~2개 제거"). 계획 승인 조건에 따라 Collector 와 같게 쓴다.
            k = int(rng.integers(1, 3))
            unfastened = sorted(int(m) + 1 for m in rng.choice(BOLT_COUNT, size=k, replace=False))
        image = make_image(rng, unfastened, drift, strength if drift else 0.0)
        validate_image_array(image)  # 128x128, uint8 (계약 1장)
        images.append(image)
        parts.append(unfastened)
    return Cell(np.stack(images), labels, parts)


def manifest_hashes(root) -> set:
    lines = (Path(root) / DATASET_DIR / MANIFEST).read_text(encoding="utf-8").splitlines()
    return {json.loads(line)["image_md5"] for line in lines}


def check_sweep_overlap(cell_hashes: dict, dataset_hashes: set) -> None:
    """스윕 이미지는 서로, 그리고 data/dataset 의 어떤 이미지와도 같으면 안 된다."""
    check_no_overlap({**cell_hashes, "data/dataset": dataset_hashes})


# ---------- 측정 ----------

def cell_metrics(predictor, cell: Cell) -> dict:
    m = evaluate(predictor, cell.images, cell.labels, predictor.threshold)
    return {"accuracy": m.accuracy, "recall": m.recall, "false_alarm": false_alarm_rate(m.confusion_matrix),
            "cm": m.confusion_matrix, "n": m.n}


def _train_v1(tmp, data, seed, predictor_config):
    # train() 은 평가셋을 요구해서 기존 test_normal / test_drift 를 넘긴다. 스윕 결과에는 쓰지 않는다.
    train(data["train"].images, data["train"].labels, version="v1",
          eval_normal=(data["test_normal"].images, data["test_normal"].labels),
          eval_drift=(data["test_drift"].images, data["test_drift"].labels),
          snapshot={"train": len(data["train"].labels), "pool": 0}, seed=seed, config_path=predictor_config, root=tmp)
    predictor = Predictor(config_path=predictor_config, root=tmp)
    predictor.load("v1")
    return predictor


# ---------- 보고서 ----------

def _f(x):
    return f"{x:.4f}"


def _s(x):
    return f"{x:.3f}"


def _label(kind, s, current):
    mark = "현재값, " if kind in current and math.isclose(current[kind], s) else ""
    return f"{kind}, {_s(s)} ({mark}{describe(kind, s)})"


def _tag(accuracy):
    return "목표구간 안" if in_target(accuracy) else "목표구간 밖"


def _metric_text(m):
    return (f"정확도 {_f(m['accuracy'])}, 재현율 {_f(m['recall'])}, 오경보율 {_f(m['false_alarm'])}, "
            f"혼동행렬 [[TN, FP], [FN, TP]] {m['cm']}")


def build_report(results, strengths, info, predictor_config, n_normal, n_defect) -> str:
    seeds = list(results)
    current = info["collector"]["drift_strength"]
    cfg = load_predictor_config(predictor_config)
    lines = [
        "== 설정 ==",
        f"v1 학습 데이터: train {info['splits']['train']['size']}장 ({DATASET_DIR}/train)",
        f"학습 시드: {', '.join(map(str, seeds))}",
        f"평가 이미지: 종류·강도마다 정상 {n_normal}장 + 불량 {n_defect}장 (평가 시드 영역 {SWEEP_SEED}, data/dataset 과 해시 겹침 0)",
        f"목표 구간(정확도): {TARGET[0]:.2f}~{TARGET[1]:.2f}",
        f"collector.toml 현재 강도(읽기만, 수정 안 함): {current}",
        f"학습 epochs: {cfg.epochs}", f"학습 batch_size: {cfg.batch_size}",
        f"학습 learning_rate: {cfg.learning_rate}", f"임계값: {cfg.threshold}",
        "",
        "== 정상 환경 (드리프트 없음, 비교용) ==",
    ]
    for seed in seeds:
        m = results[seed][(BASELINE, 0.0)]
        lines.append(f"{BASELINE}, 0.000, 시드 {seed}: {_metric_text(m)}")
    for kind in DRIFT_TYPES:
        lines += ["", f"== {kind} =="]
        for s in strengths[kind]:
            name = _label(kind, s, current)
            for seed in seeds:
                m = results[seed][(kind, s)]
                lines.append(f"{name}, 시드 {seed}: {_metric_text(m)}, {_tag(m['accuracy'])}")
            mean = {k: sum(results[sd][(kind, s)][k] for sd in seeds) / len(seeds)
                    for k in ("accuracy", "recall", "false_alarm")}
            lines.append(f"{name}, 평균: 정확도 {_f(mean['accuracy'])}, 재현율 {_f(mean['recall'])}, "
                         f"오경보율 {_f(mean['false_alarm'])}, {_tag(mean['accuracy'])}")
    lines += ["", f"== 목표 구간 요약 (정확도 {TARGET[0]:.2f}~{TARGET[1]:.2f}) =="]
    for kind in DRIFT_TYPES:
        mean_in, all_in = [], []
        for s in strengths[kind]:
            accs = [results[sd][(kind, s)]["accuracy"] for sd in seeds]
            if in_target(sum(accs) / len(accs)):
                mean_in.append(_s(s))
            if all(in_target(a) for a in accs):
                all_in.append(_s(s))
        lines.append(f"{kind}, 평균 정확도가 목표 구간 안인 강도: {', '.join(mean_in) or '없음'}")
        lines.append(f"{kind}, 모든 시드의 정확도가 목표 구간 안인 강도: {', '.join(all_in) or '없음'}")
    return "\n".join(lines)


def run_sweep(root=PROJECT_ROOT, seeds=(42, 44), strengths=None, n_normal=N_NORMAL, n_defect=N_DEFECT,
              predictor_config=None, report_path=None, log=print) -> dict:
    for name, n in (("n_normal", n_normal), ("n_defect", n_defect)):
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"{name}: 1 이상의 정수여야 합니다 (받은 값: {n!r})")
    root = Path(root)
    strengths = {k: tuple((strengths or DEFAULT_STRENGTHS)[k]) for k in DRIFT_TYPES}
    data, info = load_dataset(root)

    log("평가 이미지 만드는 중...")
    cells = {(BASELINE, 0.0): make_cell(BASELINE, 0.0, n_normal, n_defect)}
    for kind in DRIFT_TYPES:
        for s in strengths[kind]:
            cells[(kind, s)] = make_cell(kind, s, n_normal, n_defect)
    check_sweep_overlap(
        {f"{k},{s}": {hashlib.md5(im.tobytes()).hexdigest() for im in c.images} for (k, s), c in cells.items()},
        manifest_hashes(root))

    results = {}
    for seed in seeds:
        log(f"시드 {seed}: v1 학습·평가 중...")
        with tempfile.TemporaryDirectory() as tmp:  # 모델은 여기에만 저장되고 끝나면 지워진다
            predictor = _train_v1(tmp, data, seed, predictor_config)
            results[seed] = {key: cell_metrics(predictor, cell) for key, cell in cells.items()}

    report = build_report(results, strengths, info, predictor_config, n_normal, n_defect)
    report_path = Path(report_path) if report_path is not None else root / "data" / REPORT_NAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")
    return {"results": results, "report": report, "report_path": report_path}


def resolve_report_path(root, text: str) -> Path:
    """--report 경로를 프로젝트 루트 기준으로 해석한다. 루트 밖이면 거부한다."""
    root = Path(root).resolve()
    path = Path(text)
    path = (path if path.is_absolute() else root / path).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError(f"--report: 프로젝트 루트({root}) 안의 경로여야 합니다 (받은 값: {text!r})") from None
    return path


def main(argv=None, root=PROJECT_ROOT) -> int:
    parser = argparse.ArgumentParser(description="드리프트 강도별로 v1 정확도·재현율·오경보율을 측정해 data/sweep_report.txt 에 쓴다")
    parser.add_argument("--seeds", default="42,44", help="v1 학습 시드 (쉼표로 구분, 기본값 42,44)")
    for kind in DRIFT_TYPES:
        parser.add_argument(f"--{kind}", default=None, help=f"{kind} 강도 목록 (기본값 {','.join(map(str, DEFAULT_STRENGTHS[kind]))})")
    parser.add_argument("--n-normal", type=int, default=N_NORMAL, help=f"칸마다 정상 평가 이미지 수 (기본값 {N_NORMAL})")
    parser.add_argument("--n-defect", type=int, default=N_DEFECT, help=f"칸마다 불량 평가 이미지 수 (기본값 {N_DEFECT})")
    parser.add_argument("--report", default=None,
                        help=f"보고서 저장 경로 (프로젝트 루트 안만 허용, 기본값 data/{REPORT_NAME})")
    args = parser.parse_args(argv)
    try:
        strengths = {k: parse_strengths(getattr(args, k), k) if getattr(args, k) else DEFAULT_STRENGTHS[k]
                     for k in DRIFT_TYPES}
        report_path = resolve_report_path(root, args.report) if args.report else None
        out = run_sweep(root, seeds=parse_seeds(args.seeds), strengths=strengths, n_normal=args.n_normal,
                        n_defect=args.n_defect, report_path=report_path)
    except (DatasetError, ValueError) as e:
        print(f"실패: {e}", file=sys.stderr)
        return 1
    print(out["report"])
    print(f"\n보고서 저장: {out['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
