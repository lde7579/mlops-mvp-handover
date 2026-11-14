"""성능 저하 감지 N·T 측정 (측정만): 정답/오답 열을 오프라인으로 재생해 (N, T) 격자를 평가하고 data/detection_report.txt 에 쓴다.

실행: python -m scripts.measure_detection [--sensitivity-seeds 43,44] [--streams 20]

- 서버·DB 는 쓰지 않는다. models/v1 은 읽기만 하고, 그 밖의 모델은 임시 폴더에서 학습해 메모리로 올린 뒤 폴더를 지운다.
- 스트림 이미지는 make_image() 로 새로 만든다 (고정 시드 9101~, 풀 9201~). data/dataset 과 이미지가 겹치면 실패한다.
- 경고 규칙은 backend/monitoring.py 와 같다: 최신 N건이 모이기 전에는 보류, 모이면 correct/N < T 이면 경고.
- 합격 판정은 v1 시드 42(models/v1) 결과로만 한다. 시드 43·44 는 참고다.
- 보고서에는 숫자와 합격/불합격만 쓰고 해석은 쓰지 않는다. collector.toml, parts/, docs/ 는 수정하지 않는다.
"""
import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np

from common.schemas import validate_image_array
from parts.collector.config import load_config as load_collector_config
from parts.collector.simulator import BOLT_COUNT, DRIFT_TYPES, make_image
from parts.predictor import Predictor, train
from scripts.make_dataset import PROJECT_ROOT, DatasetError, check_no_overlap, load_dataset
from scripts.measure_baseline import parse_seeds
from scripts.sweep_drift import manifest_hashes

REPORT_NAME = "detection_report.txt"
GRID_N = (10, 15, 20, 30, 50)
GRID_T = (0.90, 0.93, 0.95, 0.96, 0.97, 0.98)
PRIMARY_SEED = 42            # 합격 판정은 models/v1 (시드 42) 결과로만 한다
STREAM_SEED_BASE = 9100      # 스트림 i 의 시드 = 9101 + i. 데이터셋(1001~1004), 스윕(9001)과 다른 영역
POOL_SEED_BASE = 9201        # 종류별 풀 시드 = 9201 + 종류 번호
KIND_INDEX = {"dark": 0, "noise": 1, "shift": 2, "normal": 3}
DEFECT_RATE = 0.2
N_STREAMS = 20
NORMAL_LEN = 600             # 정상 스트림 길이
WARMUP = 100                 # 드리프트 스트림: 정상 100건 뒤 드리프트
DRIFT_LEN = 200
POOL_SIZE = 500
DEMO_LEN = 150               # 시연 5분 = 2초 간격 150건
AFTER_LEN = 150              # 회복 뒤 재점등을 지켜보는 건수
INTERVALS = (1, 2, 3)        # 자동 생성 간격(초), api-v1.md A23
POLL_SECONDS = 5             # 대시보드 폴링 최대 지연(초), screens.md
TOLERANCE = 1e-9

# 합격 기준 (승인된 안). 판정은 시드 42 결과로만 한다.
PASS_RATE = 0.95
DETECT_MEDIAN_SECONDS = 60   # 간격 2초 기준
DETECT_INTERVAL = 2
SELECTION_RULE = "헛경고, dark 탐지, dark 회복 합격을 모두 만족하는 (N, T) 중 N이 가장 큰 것, 같으면 T가 낮은 것"

# 규칙 개정 (측정 결과를 확인한 뒤에 한 것. 원래 규칙과 그 결과는 위 그대로 보고서에 남긴다)
REVISED_RECOVERY_SECONDS = 60          # 개정 1: dark 회복 2초 간격 60초 이내 (합격 칸에서는 N <= 30)
REVISED_MIN_K = 2                      # 개정 2: 오답 1건에 경고가 켜지지 않는 칸(k >= 2)만
PREVIOUS_RECOMMENDATION = (30, 0.95)   # backend/monitoring.py 의 확정 전 권장 수치


# ---------- 경고 재생 (backend/monitoring.py drift() 와 같은 규칙) ----------

def window_alarm(correct, n: int, t: float) -> np.ndarray:
    """스텝마다 경고가 켜져 있는가. 최신 n건이 모이기 전(앞 n-1칸)은 보류라 False 로 둔다."""
    c = np.asarray(correct, dtype=np.int64)
    active = np.zeros(len(c), dtype=bool)
    if len(c) >= n:
        sums = np.concatenate([[0], np.cumsum(c)])
        active[n - 1:] = (sums[n:] - sums[:-n]) / n < t  # 서버와 같은 나눗셈·같은 부등호, 허용오차 없음
    return active


def alarm_error_count(n: int, t: float):
    """창 안 오답이 몇 건부터 경고가 켜지는가 (n건 중). 켜질 수 없으면 None."""
    for errors in range(n + 1):
        if (n - errors) / n < t:
            return errors
    return None


def false_alarm_stats(normal_streams, n: int, t: float) -> dict:
    """정상 스트림에서의 헛경고. quiet = 시연 150건 동안 한 번도 안 켜진 스트림 비율, step_rate = 창이 찬 뒤 켜진 시점 비율."""
    series = [window_alarm(s, n, t) for s in normal_streams]
    quiet = sum(not a[:DEMO_LEN].any() for a in series)
    on = sum(int(a[n - 1:].sum()) for a in series)
    total = sum(len(a) - (n - 1) for a in series)
    return {"quiet_rate": quiet / len(series), "step_rate": on / total, "streams": len(series)}


def detect_count(correct, n: int, t: float, warmup: int = WARMUP):
    """드리프트 주입 뒤 경고가 처음 켜져 있는 시점까지의 판정 건수 (주입 직후 1건째 = 1). 켜지지 않으면 None."""
    active = window_alarm(correct, n, t)
    hits = np.flatnonzero(active[warmup:])
    return int(hits[0]) + 1 if len(hits) else None


def recover_info(correct_v2, n: int, t: float, after: int = AFTER_LEN) -> dict:
    """v2 로 바뀐 뒤: 창은 v2 판정 0건에서 다시 시작한다. n건이 찬 시점부터 after 건 동안을 본다.

    count = 그 뒤로 끝까지 꺼져 있게 되는 첫 시점의 판정 건수 (n 이면 채워지자마자 꺼져 있고 재점등도 없음). 없으면 None.
    """
    if len(correct_v2) < n + after:
        raise ValueError(f"v2 판정이 {n + after}건 이상 필요합니다 (받은 건수: {len(correct_v2)})")
    horizon = window_alarm(correct_v2, n, t)[n - 1:n + after]
    count = None
    if not horizon[-1]:
        on = np.flatnonzero(horizon)
        count = n + (int(on[-1]) + 1 if len(on) else 0)
    return {"count": count, "pass": count == n, "relit": bool((~horizon[:-1] & horizon[1:]).any())}


def median_or_none(values):
    """None(= 해당 없음)을 무한대로 보고 가운데 값을 구한다. 가운데가 무한대면 None."""
    ordered = sorted(float("inf") if v is None else v for v in values)
    mid = ordered[(len(ordered) - 1) // 2] if len(ordered) % 2 else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
    return None if mid == float("inf") else mid


def evaluate_cell(streams: dict, v2_streams: dict, n: int, t: float) -> dict:
    """(N, T) 한 칸. streams: {"normal": [...], kind: [...]}, v2_streams: {kind: [...]} (정답 여부 불리언 열)."""
    cell = {"k": alarm_error_count(n, t), "false_alarm": false_alarm_stats(streams["normal"], n, t), "detect": {}, "recover": {}}
    for kind in DRIFT_TYPES:
        counts = [detect_count(s, n, t) for s in streams[kind]]
        cell["detect"][kind] = {
            "rate": sum(c is not None and c <= n for c in counts) / len(counts),
            "any_rate": sum(c is not None for c in counts) / len(counts),
            "median_count": median_or_none(counts)}
        infos = [recover_info(s, n, t) for s in v2_streams[kind]]
        cell["recover"][kind] = {
            "pass_rate": sum(i["pass"] for i in infos) / len(infos),
            "relit_rate": sum(i["relit"] for i in infos) / len(infos),
            "median_count": median_or_none([i["count"] for i in infos])}
    return cell


def judge_cell(cell: dict, n: int) -> dict:
    """합격 3항목 (헛경고, dark 탐지, dark 회복). noise·shift 는 판정하지 않는다."""
    detect, recover = cell["detect"]["dark"], cell["recover"]["dark"]
    median = detect["median_count"]
    fast = median is not None and median * DETECT_INTERVAL <= DETECT_MEDIAN_SECONDS + TOLERANCE
    result = {"false_alarm": cell["false_alarm"]["quiet_rate"] >= PASS_RATE - TOLERANCE,
              "dark_detect": detect["rate"] >= PASS_RATE - TOLERANCE and fast,
              "dark_recover": recover["pass_rate"] >= PASS_RATE - TOLERANCE}
    result["all"] = all(result.values())
    return result


def evaluate_grid(streams, v2_streams, grid_n=GRID_N, grid_t=GRID_T) -> dict:
    return {(n, t): evaluate_cell(streams, v2_streams, n, t) for n in grid_n for t in grid_t}


def select_combination(judged: dict):
    """합격 3항목을 모두 만족하는 (N, T) 중 N 이 가장 큰 것, 같으면 T 가 낮은 것. 없으면 None."""
    passed = [key for key, j in judged.items() if j["all"]]
    return max(passed, key=lambda key: (key[0], -key[1])) if passed else None


def select_revised(grid: dict, verdicts: dict):
    """개정 규칙: 조건을 필터로 먼저 적용한 뒤 N 최대 -> dark 탐지 중앙값 최소 -> T 최소 순으로 고른다. 후보가 없으면 None.

    필터: 원래 합격 3항목 통과, dark 회복 중앙값 x 2초 <= 60초, k >= 2 (오답 1건에 경고가 켜지지 않음).
    """
    candidates = []
    for key, cell in grid.items():
        recover = cell["recover"]["dark"]["median_count"]
        if not verdicts[key]["all"] or recover is None or cell["k"] is None:
            continue
        if recover * DETECT_INTERVAL > REVISED_RECOVERY_SECONDS + TOLERANCE or cell["k"] < REVISED_MIN_K:
            continue
        candidates.append((-key[0], cell["detect"]["dark"]["median_count"], key[1], key))
    return min(candidates)[3] if candidates else None


# ---------- 이미지 ----------

def make_sequence(seed_parts, n_pre: int, n_drift: int, kind, strength: float):
    """정상 n_pre 장 뒤에 kind 드리프트 n_drift 장 (kind 가 None 이면 모두 정상). 불량은 약 20%, 규칙은 Collector 와 같다."""
    rng = np.random.default_rng(np.random.SeedSequence(list(seed_parts)))
    images, labels = [], []
    for i in range(n_pre + n_drift):
        label = "NG" if rng.random() < DEFECT_RATE else "OK"
        unfastened = []
        if label == "NG":
            k = int(rng.integers(1, 3))  # 1~2개 미체결 (collector.py next_sample 과 같은 규칙)
            unfastened = sorted(int(m) + 1 for m in rng.choice(BOLT_COUNT, size=k, replace=False))
        drift = kind if kind is not None and i >= n_pre else None
        image = make_image(rng, unfastened, drift, strength if drift else 0.0)
        validate_image_array(image)
        images.append(image)
        labels.append(label)
    return np.stack(images), labels


def _md5s(images) -> set:
    return {hashlib.md5(im.tobytes()).hexdigest() for im in images}


def fingerprint(root) -> dict:
    """읽기 전용 파일의 sha256: data/mlops.db(-wal, -shm), models/v1/*. 없는 파일은 건너뛴다."""
    root = Path(root)
    paths = [root / "data" / name for name in ("mlops.db", "mlops.db-wal", "mlops.db-shm")]
    models = root / "models" / "v1"
    if models.is_dir():
        paths += sorted(p for p in models.iterdir() if p.is_file())
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


# ---------- 모델 ----------

def _fit(data, extra, seed, predictor_config, version):
    """임시 폴더에 학습하고 메모리로 올린 뒤 폴더를 지운다. extra 가 있으면 train 에 더해 학습한다."""
    images, labels = data["train"].images, list(data["train"].labels)
    snapshot = {"train": len(labels), "pool": 0}
    if extra is not None:
        images, labels = np.concatenate([images, extra[0]]), labels + list(extra[1])
        snapshot["pool"] = len(extra[1])
    with tempfile.TemporaryDirectory() as tmp:
        # train() 이 평가셋을 요구해서 기존 test_* 를 넘긴다 (평가만 하고 학습에는 쓰지 않는다)
        train(images, labels, version=version,
              eval_normal=(data["test_normal"].images, data["test_normal"].labels),
              eval_drift=(data["test_drift"].images, data["test_drift"].labels),
              snapshot=snapshot, seed=seed, config_path=predictor_config, root=tmp)
        predictor = Predictor(config_path=predictor_config, root=tmp)
        predictor.load(version)
    return predictor


def _correct(predictor, images, labels) -> np.ndarray:
    out = []
    for image, label in zip(images, labels):
        got = predictor.predict(image).predicted_label
        out.append(getattr(got, "value", got) == label)
    return np.array(out, dtype=bool)


# ---------- 측정 ----------

def run_measure(root=PROJECT_ROOT, sensitivity_seeds=(43, 44), n_streams=N_STREAMS, predictor_config=None,
                report_path=None, log=print) -> dict:
    root = Path(root)
    start_print = fingerprint(root)
    data, info = load_dataset(root)
    strength = load_collector_config().drift_strength

    log("풀·스트림 이미지 만드는 중...")
    pools = {k: make_sequence([POOL_SEED_BASE + KIND_INDEX[k]], 0, POOL_SIZE, k, strength[k]) for k in DRIFT_TYPES}
    streams = {"normal": []}
    for kind in DRIFT_TYPES:
        streams[kind] = []
    for i in range(n_streams):
        seed = STREAM_SEED_BASE + 1 + i
        streams["normal"].append(make_sequence([seed, KIND_INDEX["normal"]], NORMAL_LEN, 0, None, 0.0))
        for kind in DRIFT_TYPES:
            streams[kind].append(make_sequence([seed, KIND_INDEX[kind]], WARMUP, DRIFT_LEN, kind, strength[kind]))
    check_no_overlap({
        "streams": set().union(*[_md5s(images) for group in streams.values() for images, _ in group]),
        **{f"pool_{k}": _md5s(p[0]) for k, p in pools.items()},
        "data/dataset": manifest_hashes(root)})

    seeds = (PRIMARY_SEED,) + tuple(s for s in sensitivity_seeds if s != PRIMARY_SEED)
    log(f"모델 준비 중 (시드 {', '.join(map(str, seeds))})...")
    v1 = {}
    primary = Predictor(config_path=predictor_config, root=root)
    primary.load("v1")  # models/v1 읽기 전용
    v1[PRIMARY_SEED] = primary
    v2 = {}
    for seed in seeds:
        if seed != PRIMARY_SEED:
            log(f"시드 {seed}: v1 학습 중...")
            v1[seed] = _fit(data, None, seed, predictor_config, "v1")
        for kind in DRIFT_TYPES:
            log(f"시드 {seed}: v2({kind} 풀) 학습 중...")
            v2[(seed, kind)] = _fit(data, pools[kind], seed, predictor_config, "v2")

    log("판정 중...")
    judged = {}
    for seed in seeds:
        v1_streams = {"normal": [_correct(v1[seed], im, lab) for im, lab in streams["normal"]]}
        v2_streams = {}
        for kind in DRIFT_TYPES:
            v1_streams[kind] = [_correct(v1[seed], im, lab) for im, lab in streams[kind]]
            v2_streams[kind] = [_correct(v2[(seed, kind)], im[WARMUP:], lab[WARMUP:]) for im, lab in streams[kind]]
        judged[seed] = {"v1": v1_streams, "v2": v2_streams}

    grids = {seed: evaluate_grid(j["v1"], j["v2"]) for seed, j in judged.items()}
    verdicts = {key: judge_cell(cell, key[0]) for key, cell in grids[PRIMARY_SEED].items()}
    chosen = select_combination(verdicts)
    revised = select_revised(grids[PRIMARY_SEED], verdicts)

    end_print = fingerprint(root)
    report = build_report(grids, verdicts, chosen, judged, info, strength, seeds, n_streams, start_print, end_print)
    report_path = Path(report_path) if report_path is not None else root / "data" / REPORT_NAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")
    return {"grids": grids, "verdicts": verdicts, "chosen": chosen, "revised": revised, "report": report, "report_path": report_path,
            "unchanged": start_print == end_print}


# ---------- 보고서 (숫자와 합격/불합격만) ----------

def _pct(x):
    return f"{x * 100:.1f}%"


def _secs(count):
    """건수 -> '건수 (1초 간격 X초, 2초 간격 Y초, 3초 간격 Z초)'. 없으면 '없음'."""
    if count is None:
        return "없음"
    return f"{count:g}건 (" + ", ".join(f"{i}초 간격 {count * i:g}초" for i in INTERVALS) + ")"


def _ok(flag):
    return "합격" if flag else "불합격"


def _mean_acc(arrays, start=0):
    return float(np.mean(np.concatenate([a[start:] for a in arrays])))


def _cell_lines(prefix, n, t, cell, verdict=None):
    lines = [f"{prefix}N={n} T={t:.2f} (경고가 켜지는 창 안 오답 수 k={cell['k']})"]
    fa = cell["false_alarm"]
    text = f"  헛경고: 시연 {DEMO_LEN}건 동안 안 켜진 스트림 {_pct(fa['quiet_rate'])}, 창이 찬 뒤 켜진 시점 {_pct(fa['step_rate'])}"
    lines.append(text + (f", {_ok(verdict['false_alarm'])}" if verdict else ""))
    for kind in DRIFT_TYPES:
        d, r = cell["detect"][kind], cell["recover"][kind]
        mark_d = f", {_ok(verdict['dark_detect'])}" if verdict and kind == "dark" else ""
        mark_r = f", {_ok(verdict['dark_recover'])}" if verdict and kind == "dark" else ""
        lines.append(f"  (가) 탐지 {kind}: 주입 후 N건 이내 탐지 {_pct(d['rate'])}, 끝까지 탐지 {_pct(d['any_rate'])}, "
                     f"중앙값 {_secs(d['median_count'])}, 폴링 지연 최대 +{POLL_SECONDS}초 별도{mark_d}")
        lines.append(f"  (나) 회복 {kind}: 채워지자마자 꺼지고 재점등 없음 {_pct(r['pass_rate'])}, 꺼진 뒤 재점등 {_pct(r['relit_rate'])}, "
                     f"회복 중앙값 {_secs(r['median_count'])}{mark_r}")
    return lines


def _normal_wrong(judged_seed) -> tuple:
    """정상 스트림에서 v1 의 오답 건수와 전체 건수."""
    arrays = [np.asarray(a, dtype=bool) for a in judged_seed["v1"]["normal"]]
    total = sum(len(a) for a in arrays)
    return total - int(sum(a.sum() for a in arrays)), total


def _revision_lines(grids, verdicts, chosen, judged, seeds) -> list:
    """규칙 개정 절. 위의 원래 규칙과 선택 결과는 그대로 두고, 개정과 그 결과를 계산해서 쓴다."""
    revised = select_revised(grids[PRIMARY_SEED], verdicts)
    lines = ["", "== 규칙 개정 (결과 확인 후) ==",
             "개정 1·2는 모두 측정 결과를 확인한 뒤에 한 것이다. 위의 선택 규칙(측정 전에 고정)과 선택 결과는 그대로 둔다.",
             f"개정 1: 합격 항목에 'dark 회복 {DETECT_INTERVAL}초 간격 {REVISED_RECOVERY_SECONDS}초 이내 (합격 칸에서는 N <= 30)'를 추가한다. "
             "사유: 원래 규칙에는 시간 제한이 없었고, e2e 3단계 시연 배분이 2분이다."]
    if chosen is not None:
        median = grids[PRIMARY_SEED][chosen]["recover"]["dark"]["median_count"]
        lines.append(f"원래 선택 결과 N={chosen[0]}, T={chosen[1]:.2f} 의 dark 회복은 {DETECT_INTERVAL}초 간격 "
                     f"{'없음' if median is None else f'{median * DETECT_INTERVAL:g}초'}")
    lines += [f"개정 2: 동점 처리 'T가 낮은 것'을 'k >= {REVISED_MIN_K}인 칸(오답 1건에 경고가 켜지지 않음) 중 dark 탐지 중앙값이 가장 작은 T'로 바꾼다. "
              "남는 동점은 낮은 T를 택한다. 사유: 시드 42는 T가 헛경고에 영향을 주지 못해 'T가 낮은 것'에 근거가 없다 (아래).",
              "개정 규칙 정렬: 조건(원래 합격 3항목, dark 회복 시간, k)을 필터로 먼저 적용한 뒤 N 최대, dark 탐지 중앙값 최소, T 최소 순"]
    if revised is None:
        lines.append("개정 규칙 선택 결과: 만족하는 (N, T) 조합 없음. 기준을 바꾸지 않고 멈춰서 보고한다.")
    else:
        cell = grids[PRIMARY_SEED][revised]
        lines.append(f"개정 규칙 선택 결과: N={revised[0]}, T={revised[1]:.2f} (경고가 켜지는 창 안 오답 수 k={cell['k']})")
        lines.append(f"이전 권장 수치 N={PREVIOUS_RECOMMENDATION[0]}, T={PREVIOUS_RECOMMENDATION[1]:.2f} 와 "
                     + ("같다" if revised == PREVIOUS_RECOMMENDATION else "다르다"))
        lines.append("원래 규칙의 선택 결과와 " + ("같다" if revised == chosen else "다르다"))
    wrong, total = _normal_wrong(judged[PRIMARY_SEED])
    if wrong == 0:
        lines.append(f"시드 {PRIMARY_SEED} 정상 스트림 오답 {wrong}건 / {total}건. 오답이 0건이라 헛경고 합격은 모든 (N, T) 칸에서 "
                     "자동 통과이고 칸을 가르지 못한다")
    else:
        passed = sum(v["false_alarm"] for v in verdicts.values())
        lines.append(f"시드 {PRIMARY_SEED} 정상 스트림 오답 {wrong}건 / {total}건. 헛경고 합격을 통과한 칸 {passed}/{len(verdicts)}")
    for seed in seeds:
        if seed != PRIMARY_SEED:
            wrong, total = _normal_wrong(judged[seed])
            lines.append(f"참고(합격 판정에 쓰지 않음): 시드 {seed} 정상 스트림 오답 {wrong}건 / {total}건")
    return lines


def build_report(grids, verdicts, chosen, judged, info, strength, seeds, n_streams, start_print, end_print) -> str:
    lines = [
        "== 선택 규칙 (측정 전에 고정) ==",
        f"조합 선택 규칙: {SELECTION_RULE}",
        "만족하는 조합이 없으면 기준을 바꾸지 않고 멈춰서 보고한다.",
        f"합격 판정은 v1 시드 {PRIMARY_SEED} (models/v1) 결과로만 한다. 다른 시드는 참고다.",
        f"헛경고 합격: 시연 {DEMO_LEN}건(2초 간격 5분) 동안 경고가 한 번도 안 켜진 정상 스트림 비율 >= {_pct(PASS_RATE)}",
        f"dark 탐지 합격: 주입 후 N건 이내에 경고가 켜진 스트림 >= {_pct(PASS_RATE)} 이고, 중앙값 시간이 {DETECT_INTERVAL}초 간격에서 {DETECT_MEDIAN_SECONDS}초 이하",
        f"dark 회복 합격: v2 교체 후 N건이 찬 시점에 경고가 꺼져 있고 이후 {AFTER_LEN}건 재점등이 없는 스트림 >= {_pct(PASS_RATE)}",
        "noise, shift: 합격 기준 없음 (보고만). shift 는 회복이 안 되는 것이 예상 정상 결과",
        "",
        "== 선택 결과 ==",
    ]
    if chosen is None:
        lines.append("만족하는 (N, T) 조합 없음")
    else:
        lines.append(f"선택된 조합: N={chosen[0]}, T={chosen[1]:.2f}")
    lines.append("합격 3항목을 모두 만족한 조합: " + (", ".join(f"N={n} T={t:.2f}" for (n, t), v in verdicts.items() if v["all"]) or "없음"))
    lines += _revision_lines(grids, verdicts, chosen, judged, seeds)

    lines += ["", "== 설정 ==",
              f"스트림: 종류(정상, dark, noise, shift)마다 {n_streams}개, 시드 {STREAM_SEED_BASE + 1}~{STREAM_SEED_BASE + n_streams} (data/dataset 과 해시 겹침 0)",
              f"정상 스트림 {NORMAL_LEN}건, 드리프트 스트림 정상 {WARMUP}건 뒤 드리프트 {DRIFT_LEN}건, 불량 약 {int(DEFECT_RATE * 100)}%",
              f"v2 학습: train {info['splits']['train']['size']}장 + 해당 종류 드리프트 풀 {POOL_SIZE}장 (시드 {POOL_SEED_BASE}~)",
              f"드리프트 강도(collector.toml, 읽기만): {strength}",
              f"격자 N: {', '.join(map(str, GRID_N))} (100은 2초 간격에서 200초라 시연 불가로 제외)",
              f"격자 T: {', '.join(f'{t:.2f}' for t in GRID_T)}",
              f"모델 시드: 주결과 {PRIMARY_SEED} (models/v1), 참고 {', '.join(str(s) for s in seeds if s != PRIMARY_SEED) or '없음'}",
              f"자동 생성 간격 {', '.join(map(str, INTERVALS))}초, 폴링 지연 최대 +{POLL_SECONDS}초",
              "", "== 읽기 전용 파일 해시 (시작 -> 끝) =="]
    for name in sorted(set(start_print) | set(end_print)):
        a, b = start_print.get(name), end_print.get(name)
        lines.append(f"{name}: {'일치' if a == b else '불일치'} (sha256 {a})")
    lines.append(f"전체: {'일치' if start_print == end_print else '불일치'}")

    lines += ["", "== 모델별 정확도 (스트림 합산) =="]
    for seed in seeds:
        j = judged[seed]
        tag = "주결과" if seed == PRIMARY_SEED else "참고"
        lines.append(f"시드 {seed} ({tag}): v1 정상 {_mean_acc(j['v1']['normal']):.4f}")
        for kind in DRIFT_TYPES:
            lines.append(f"시드 {seed} ({tag}): {kind} v1 {_mean_acc(j['v1'][kind], WARMUP):.4f}, v2 {_mean_acc(j['v2'][kind]):.4f}")

    lines += ["", f"== 주결과: 시드 {PRIMARY_SEED} (models/v1) =="]
    for (n, t), cell in grids[PRIMARY_SEED].items():
        lines += _cell_lines("", n, t, cell, verdicts[(n, t)])
        lines.append(f"  합격 종합: {_ok(verdicts[(n, t)]['all'])}")
    for seed in seeds:
        if seed == PRIMARY_SEED:
            continue
        lines += ["", f"== 참고 (합격 판정에 사용 안 함): v1 시드 {seed} =="]
        for (n, t), cell in grids[seed].items():
            lines += _cell_lines("", n, t, cell)
    return "\n".join(lines)


def main(argv=None, root=PROJECT_ROOT) -> int:
    parser = argparse.ArgumentParser(description="(N, T) 격자를 오프라인으로 재생해 data/detection_report.txt 에 쓴다")
    parser.add_argument("--sensitivity-seeds", default="43,44", help="참고용 학습 시드 (쉼표로 구분, 기본값 43,44)")
    parser.add_argument("--streams", type=int, default=N_STREAMS, help=f"종류마다 스트림 수 (기본값 {N_STREAMS})")
    args = parser.parse_args(argv)
    try:
        if args.streams < 1:
            raise ValueError(f"--streams: 1 이상의 정수여야 합니다 (받은 값: {args.streams})")
        out = run_measure(root, sensitivity_seeds=parse_seeds(args.sensitivity_seeds), n_streams=args.streams)
    except (DatasetError, ValueError) as e:
        print(f"실패: {e}", file=sys.stderr)
        return 1
    print(f"보고서 저장: {out['report_path']}")
    print("선택된 조합: " + ("없음" if out["chosen"] is None else f"N={out['chosen'][0]}, T={out['chosen'][1]:.2f}"))
    print("개정 규칙 선택 결과: " + ("없음" if out["revised"] is None else f"N={out['revised'][0]}, T={out['revised'][1]:.2f}"))
    print("읽기 전용 파일 해시: " + ("일치" if out["unchanged"] else "불일치"))
    return 0 if out["unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
