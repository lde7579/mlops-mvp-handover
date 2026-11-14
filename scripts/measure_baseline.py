"""기준 성능 측정: v1(train)과 v2(train + pool)를 학습해 지표, G1~G3, 시간을 data/baseline_report.txt 에 쓴다.

실행: python -m scripts.measure_baseline [--seeds 42,43,44]

- parts.predictor 의 train(), evaluate(), Predictor 만 쓴다. 모델은 임시 폴더에 저장하고 끝나면 지운다.
- 드리프트 강도, 모델, 학습 설정은 바꾸지 않는다 (predictor.toml, collector.toml 그대로).
- 보고서에는 숫자와 통과/미통과만 쓰고 해석은 쓰지 않는다.
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from parts.collector.simulator import DRIFT_TYPES
from parts.predictor import Predictor, evaluate, train
from parts.predictor.config import load_config as load_predictor_config
from scripts.make_dataset import PROJECT_ROOT, DatasetError, load_dataset

TOLERANCE = 1e-9  # 0.01 같은 기준을 부동소수점 오차 때문에 잘못 판정하지 않게 한다
REPORT_NAME = "baseline_report.txt"
WARMUP = 10


def parse_seeds(text: str) -> tuple:
    try:
        seeds = tuple(int(s) for s in text.split(","))
    except ValueError:
        raise ValueError(f"--seeds: 42,43,44 처럼 쉼표로 나눈 정수여야 합니다 (받은 값: {text!r})") from None
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"--seeds: 비어 있거나 같은 값이 겹칩니다 (받은 값: {text!r})")
    return seeds


def _kind_metrics(predictor, split, kind):
    mask = np.array([k == kind for k in split.kinds])
    if not mask.any():
        return None
    labels = [lab for lab, m in zip(split.labels, mask) if m]
    return evaluate(predictor, split.images[mask], labels, predictor.threshold)


def _measure_version(tmp, predictor_config, data, version, images, labels, snapshot, seed):
    start = time.perf_counter()
    metrics = train(images, labels, version=version,
                    eval_normal=(data["test_normal"].images, data["test_normal"].labels),
                    eval_drift=(data["test_drift"].images, data["test_drift"].labels),
                    snapshot=snapshot, seed=seed, config_path=predictor_config, root=tmp)
    seconds = time.perf_counter() - start
    predictor = Predictor(config_path=predictor_config, root=tmp)
    predictor.load(version)
    kinds = {k: _kind_metrics(predictor, data["test_drift"], k) for k in DRIFT_TYPES}
    return {"normal": metrics.normal, "drift": metrics.drift, "kinds": kinds, "train_sec": seconds}, predictor


def _inference_ms(predictor, images):
    for _ in range(WARMUP):
        predictor.predict(images[0])
    times = []
    for image in images:
        start = time.perf_counter()
        predictor.predict(image)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.mean(times)), float(np.max(times))


def gates(v1, v2):
    """G1~G3. diff 는 v2 - v1, need 는 통과에 필요한 최소 차이."""
    spec = {
        "G1": (v2["drift"].accuracy - v1["drift"].accuracy, 0.01),
        "G2": (v2["drift"].recall - v1["drift"].recall, -0.01),
        "G3": (v2["normal"].accuracy - v1["normal"].accuracy, -0.01),
    }
    return {g: {"diff": d, "need": n, "passed": d >= n - TOLERANCE} for g, (d, n) in spec.items()}


def run_seed(data, seed, predictor_config):
    n_train, n_pool = len(data["train"].labels), len(data["pool"].labels)
    with tempfile.TemporaryDirectory() as tmp:  # 모델은 여기에만 저장되고 끝나면 지워진다
        v1, predictor_v1 = _measure_version(
            tmp, predictor_config, data, "v1", data["train"].images, data["train"].labels,
            {"train": n_train, "pool": 0}, seed)
        mean_ms, max_ms = _inference_ms(predictor_v1, data["test_normal"].images)
        v2, _ = _measure_version(
            tmp, predictor_config, data, "v2",
            np.concatenate([data["train"].images, data["pool"].images]),
            data["train"].labels + data["pool"].labels, {"train": n_train, "pool": n_pool}, seed)
    return {"seed": seed, "v1": v1, "v2": v2, "gates": gates(v1, v2),
            "infer_mean_ms": mean_ms, "infer_max_ms": max_ms}


# ---------- 보고서 (숫자와 통과/미통과만) ----------

def _f(x):
    return f"{x:.4f}"


def _metric_lines(prefix, m):
    if m is None:
        return [f"{prefix} 건수: 0"]
    return [f"{prefix} 정확도: {_f(m.accuracy)}", f"{prefix} 재현율: {_f(m.recall)}",
            f"{prefix} 혼동행렬 [[TN, FP], [FN, TP]]: {m.confusion_matrix}", f"{prefix} 건수: {m.n}"]


def _seed_lines(r):
    lines = [f"== 시드 {r['seed']} =="]
    for ver in ("v1", "v2"):
        lines += _metric_lines(f"{ver} test_normal", r[ver]["normal"])
        lines += _metric_lines(f"{ver} test_drift", r[ver]["drift"])
        for kind in DRIFT_TYPES:
            lines += _metric_lines(f"{ver} test_drift {kind}", r[ver]["kinds"][kind])
    names = {"G1": "test_drift 정확도 v2 >= v1 +1%p", "G2": "test_drift 재현율 v2 >= v1 -1%p",
             "G3": "test_normal 정확도 v2 >= v1 -1%p"}
    for g, v in r["gates"].items():
        lines.append(f"{g} ({names[g]}): {'통과' if v['passed'] else '미통과'} "
                     f"(차이 {v['diff'] * 100:+.2f}%p, 기준 {v['need'] * 100:+.2f}%p 이상)")
    lines += [f"v1 학습 시간(초, train() 호출 전체): {r['v1']['train_sec']:.1f}",
              f"v2 학습 시간(초, train() 호출 전체): {r['v2']['train_sec']:.1f}",
              f"이미지 1장 추론 시간 평균(ms): {r['infer_mean_ms']:.2f}",
              f"이미지 1장 추론 시간 최대(ms): {r['infer_max_ms']:.2f}"]
    return lines


def _mean(values):
    return sum(values) / len(values)


def _average_lines(results):
    lines = [f"== 시드 평균 (시드 {len(results)}개: {', '.join(str(r['seed']) for r in results)}) =="]

    def pick(r, ver, where, attr):
        m = r[ver]["kinds"][where] if where in DRIFT_TYPES else r[ver][where]
        return getattr(m, attr) if m is not None else None

    for ver in ("v1", "v2"):
        for where, label in (("normal", "test_normal"), ("drift", "test_drift"), *[(k, f"test_drift {k}") for k in DRIFT_TYPES]):
            for attr, name in (("accuracy", "정확도"), ("recall", "재현율")):
                vals = [pick(r, ver, where, attr) for r in results]
                if None not in vals:
                    lines.append(f"{ver} {label} {name} 평균: {_f(_mean(vals))}")
    for g in ("G1", "G2", "G3"):
        lines.append(f"{g} 차이 평균: {_mean([r['gates'][g]['diff'] for r in results]) * 100:+.2f}%p")
        lines.append(f"{g} 통과한 시드 수: {sum(r['gates'][g]['passed'] for r in results)}/{len(results)}")
    lines += [f"v1 학습 시간 평균(초): {_mean([r['v1']['train_sec'] for r in results]):.1f}",
              f"v2 학습 시간 평균(초): {_mean([r['v2']['train_sec'] for r in results]):.1f}",
              f"이미지 1장 추론 시간 평균(ms): {_mean([r['infer_mean_ms'] for r in results]):.2f}",
              f"이미지 1장 추론 시간 최대(ms): {max(r['infer_max_ms'] for r in results):.2f}"]
    return lines


def _header_lines(info, predictor_config):
    cfg = load_predictor_config(predictor_config)
    lines = ["== 설정 =="]
    for name, s in info["splits"].items():
        lines.append(f"{name}: {s['size']}장 (정상 {s['ok']}, 불량 {s['ng']}, 시드 {s['seed']})")
    lines += [f"학습 epochs: {cfg.epochs}", f"학습 batch_size: {cfg.batch_size}",
              f"학습 learning_rate: {cfg.learning_rate}", f"임계값: {cfg.threshold}",
              f"드리프트 강도: {info['collector']['drift_strength']}",
              f"torch 스레드 수: {torch.get_num_threads()}"]
    return lines


def run_baseline(root=PROJECT_ROOT, seeds=(42,), predictor_config=None, report_path=None, log=print) -> dict:
    root = Path(root)
    data, info = load_dataset(root)
    results = []
    for seed in seeds:
        log(f"시드 {seed} 측정 중...")
        results.append(run_seed(data, seed, predictor_config))
    lines = _header_lines(info, predictor_config)
    for r in results:
        lines += [""] + _seed_lines(r)
    lines += [""] + _average_lines(results)
    report_path = Path(report_path) if report_path is not None else root / "data" / REPORT_NAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"results": results, "report": "\n".join(lines), "report_path": report_path}


def main(argv=None, root=PROJECT_ROOT) -> int:
    parser = argparse.ArgumentParser(description="v1/v2 기준 성능, G1~G3, 시간을 측정해 data/baseline_report.txt 에 쓴다")
    parser.add_argument("--seeds", default="42", help="학습 시드. 쉼표로 여러 개 (기본값 42, 예: 42,43,44)")
    args = parser.parse_args(argv)
    try:
        out = run_baseline(root, seeds=parse_seeds(args.seeds))
    except (DatasetError, ValueError) as e:
        print(f"실패: {e}", file=sys.stderr)
        return 1
    print(out["report"])
    print(f"\n보고서 저장: {out['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
