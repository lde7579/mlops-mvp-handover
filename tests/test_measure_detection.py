"""scripts/measure_detection.py 테스트: 가짜 정답 열로 재생 규칙·집계·선택 규칙을 확인한다 (학습·큰 이미지 생성 없음)."""
import numpy as np
import pytest

from parts.collector.simulator import DRIFT_TYPES
from scripts.make_dataset import DatasetError
from scripts.measure_detection import (
    AFTER_LEN,
    DEMO_LEN,
    GRID_N,
    GRID_T,
    SELECTION_RULE,
    WARMUP,
    alarm_error_count,
    build_report,
    detect_count,
    evaluate_cell,
    evaluate_grid,
    false_alarm_stats,
    fingerprint,
    judge_cell,
    make_sequence,
    median_or_none,
    recover_info,
    select_combination,
    select_revised,
    window_alarm,
)
from scripts.make_dataset import check_no_overlap


def seq(*parts):
    """('o', 5) = 맞음 5건, ('x', 2) = 틀림 2건."""
    return [p == "o" for kind, count in parts for p in [kind] * count]


def test_alarm_waits_for_n_judgments():
    active = window_alarm(seq(("x", 5)), 10, 0.95)
    assert not active.any()  # 10건이 모이기 전에는 아무리 틀려도 보류
    assert window_alarm(seq(("x", 10)), 10, 0.95)[9]


def test_alarm_is_strictly_below_threshold():
    # 20건 중 1건 틀림 = 정확히 0.95 -> 안 켜짐 (accuracy < T), 2건 틀림 = 0.90 -> 켜짐
    assert not window_alarm(seq(("o", 19), ("x", 1)), 20, 0.95)[19]
    assert window_alarm(seq(("o", 18), ("x", 2)), 20, 0.95)[19]


def test_alarm_error_count_matches_discrete_rule():
    assert alarm_error_count(30, 0.95) == 2 and alarm_error_count(30, 0.96) == 2
    assert alarm_error_count(30, 0.97) == 1 and alarm_error_count(30, 0.98) == 1
    assert alarm_error_count(20, 0.95) == 2 and alarm_error_count(10, 0.90) == 2


def test_alarm_turns_off_when_old_errors_leave_window():
    correct = seq(("x", 2), ("o", 20))
    active = window_alarm(correct, 10, 0.95)
    assert active[9] and not active[11 + 9]  # 오답 2건이 창 밖으로 나가면 꺼진다


def test_false_alarm_counts_only_first_demo_window():
    quiet = seq(("o", 300))
    late = seq(("o", 200), ("x", 3), ("o", 97))  # 시연 150건 뒤에 오답
    early = seq(("o", 10), ("x", 3), ("o", 287))
    stats = false_alarm_stats([quiet, late, early], 10, 0.95)
    assert stats["quiet_rate"] == pytest.approx(2 / 3)  # early 만 150건 안에 켜짐
    assert 0 < stats["step_rate"] < 1
    assert DEMO_LEN == 150


def test_detect_count_starts_at_injection():
    stream = seq(("o", WARMUP), ("o", 5), ("x", 20), ("o", 20))
    count = detect_count(stream, 10, 0.95)
    assert count == 5 + 1  # N=10 이면 오답 1건(정확도 0.90)부터 켜진다. 주입 직후 1건째 = 1
    assert detect_count(seq(("o", 300)), 10, 0.95) is None


def test_detect_count_when_alarm_already_on_at_injection():
    stream = seq(("o", WARMUP - 3), ("x", 3), ("o", 50))
    assert detect_count(stream, 10, 0.95) == 1  # 주입 직후 이미 켜져 있으면 1건째


def test_recover_passes_when_off_from_fill_and_no_relight():
    info = recover_info(seq(("o", 20 + AFTER_LEN)), 20, 0.95)
    assert info == {"count": 20, "pass": True, "relit": False}


def test_recover_counts_late_recovery_and_relit():
    late = recover_info(seq(("x", 6), ("o", 20 + AFTER_LEN)), 20, 0.95)
    assert late["count"] > 20 and not late["pass"] and not late["relit"]
    relit = recover_info(seq(("o", 40), ("x", 5), ("o", 20 + AFTER_LEN)), 20, 0.95)
    assert relit["relit"] and not relit["pass"]


def test_recover_never_off_gives_none():
    info = recover_info(seq(("x", 20 + AFTER_LEN)), 20, 0.95)
    assert info["count"] is None and not info["pass"]


def test_recover_needs_enough_judgments():
    with pytest.raises(ValueError):
        recover_info(seq(("o", 10)), 20, 0.95)


def test_window_resets_on_model_change():
    # v2 열은 v1 의 오답과 이어 붙이지 않고 따로 센다: v1 이 켜져 있어도 v2 열이 맞으면 N건째에 꺼져 있다
    v1 = window_alarm(seq(("x", 30)), 10, 0.95)
    v2 = recover_info(seq(("o", 10 + AFTER_LEN)), 10, 0.95)
    assert v1[-1] and v2["pass"]


def test_median_treats_none_as_infinite():
    assert median_or_none([3, 5, 9]) == 5
    assert median_or_none([3, None, None]) is None
    assert median_or_none([2, 4]) == 3
    assert median_or_none([2, 4, None, None]) is None


def _streams(normal, drift, v2):
    streams = {"normal": [normal] * 4}
    streams.update({k: [drift] * 4 for k in DRIFT_TYPES})
    return streams, {k: [v2] * 4 for k in DRIFT_TYPES}


def _good():
    normal = seq(("o", 300))
    drift = seq(("o", WARMUP), ("x", 15), ("o", 185))
    v2 = seq(("o", 200))
    return _streams(normal, drift, v2)


def test_evaluate_cell_and_judge_pass_when_everything_is_clean():
    streams, v2 = _good()
    cell = evaluate_cell(streams, v2, 10, 0.95)
    assert cell["false_alarm"]["quiet_rate"] == 1.0
    assert cell["detect"]["dark"]["rate"] == 1.0 and cell["detect"]["dark"]["median_count"] == 1
    assert cell["recover"]["dark"]["pass_rate"] == 1.0
    assert judge_cell(cell, 10)["all"]


def test_judge_fails_each_item_separately():
    streams, v2 = _good()
    noisy, _ = _streams(seq(("o", 10), ("x", 5), ("o", 285)), seq(("o", 300)), v2["dark"][0])
    streams_fa = {**streams, "normal": noisy["normal"]}
    assert not judge_cell(evaluate_cell(streams_fa, v2, 10, 0.95), 10)["false_alarm"]
    streams_nodetect = {**streams, "dark": [seq(("o", 300))] * 4}
    assert not judge_cell(evaluate_cell(streams_nodetect, v2, 10, 0.95), 10)["dark_detect"]
    slow = {**streams, "dark": [seq(("o", WARMUP), ("o", 40), ("x", 3), ("o", 157))] * 4}
    # 2초 간격에서 42건 = 84초 > 60초 -> 탐지 불합격 (N=50 이면 N건 이내이지만 시간이 늦다)
    assert not judge_cell(evaluate_cell(slow, v2, 50, 0.97), 50)["dark_detect"]
    bad_v2 = {**v2, "dark": [seq(("x", 30), ("o", 170))] * 4}
    assert not judge_cell(evaluate_cell(streams, bad_v2, 10, 0.95), 10)["dark_recover"]


def test_noise_and_shift_do_not_affect_judgment():
    streams, v2 = _good()
    streams = {**streams, "noise": [seq(("o", 300))] * 4, "shift": [seq(("o", 300))] * 4}
    v2 = {**v2, "noise": [seq(("x", 200))] * 4, "shift": [seq(("x", 200))] * 4}
    assert judge_cell(evaluate_cell(streams, v2, 10, 0.95), 10)["all"]


def test_select_prefers_largest_n_then_lowest_t():
    j = lambda ok: {"all": ok}  # noqa: E731
    assert select_combination({(20, 0.95): j(True), (30, 0.97): j(True), (30, 0.95): j(True), (50, 0.9): j(False)}) == (30, 0.95)
    assert select_combination({(20, 0.95): j(False)}) is None


def test_grid_contains_requested_candidates():
    assert GRID_N == (10, 15, 20, 30, 50)
    assert {0.96, 0.97, 0.98} <= set(GRID_T) and 100 not in GRID_N


def test_make_sequence_is_reproducible_and_valid():
    a, la = make_sequence([9101, 3], 3, 2, "dark", 0.18)
    b, lb = make_sequence([9101, 3], 3, 2, "dark", 0.18)
    c, _ = make_sequence([9102, 3], 3, 2, "dark", 0.18)
    assert (a == b).all() and la == lb and not (a == c).all()
    assert a.shape == (5, 128, 128) and a.dtype == np.uint8
    assert set(la) <= {"OK", "NG"}


def test_overlap_check_stops_run():
    with pytest.raises(DatasetError):
        check_no_overlap({"streams": {"a", "b"}, "data/dataset": {"b"}})


def test_fingerprint_reads_only_and_detects_change(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "models" / "v1").mkdir(parents=True)
    (tmp_path / "data" / "mlops.db").write_bytes(b"db")
    (tmp_path / "models" / "v1" / "model.pt").write_bytes(b"m")
    before = fingerprint(tmp_path)
    assert set(before) == {"data/mlops.db", "models/v1/model.pt"}  # -wal, -shm 이 없으면 건너뛴다
    assert fingerprint(tmp_path) == before
    (tmp_path / "data" / "mlops.db").write_bytes(b"db2")
    assert fingerprint(tmp_path) != before


def test_report_starts_with_fixed_rule_and_marks_reference():
    streams, v2 = _good()
    grid = evaluate_grid(streams, v2, (10,), (0.95,))
    verdicts = {k: judge_cell(c, k[0]) for k, c in grid.items()}
    judged = {42: {"v1": streams, "v2": v2}, 43: {"v1": streams, "v2": v2}}
    info = {"splits": {"train": {"size": 2000}}}
    fp = {"data/mlops.db": "ab"}
    report = build_report({42: grid, 43: grid}, verdicts, (10, 0.95), judged, info, {"dark": 0.18}, (42, 43), 4, fp, fp)
    lines = report.splitlines()
    assert lines[0] == "== 선택 규칙 (측정 전에 고정) ==" and SELECTION_RULE in lines[1]
    assert "선택된 조합: N=10, T=0.95" in report
    assert "합격 판정에 사용 안 함): v1 시드 43" in report
    assert "전체: 일치" in report and "2초 간격" in report


def fake_cell(k, detect, recover):
    """select_revised 가 보는 값만 가진 가짜 칸: k, dark 탐지 중앙값(건수), dark 회복 중앙값(건수)."""
    return {"k": k, "detect": {"dark": {"median_count": detect}}, "recover": {"dark": {"median_count": recover}}}


PASS_ALL = {"all": True}


def test_select_revised_orders_by_largest_n_then_fastest_detect_then_lowest_t():
    grid = {(20, 0.95): fake_cell(2, 3, 20), (30, 0.93): fake_cell(3, 10, 30), (30, 0.96): fake_cell(2, 5.5, 30),
            (30, 0.95): fake_cell(2, 5.5, 30)}
    assert select_revised(grid, {key: PASS_ALL for key in grid}) == (30, 0.95)  # N=30 중 탐지 5.5건이 가장 빠르고, 동점이면 낮은 T
    grid[(30, 0.96)] = fake_cell(2, 4, 30)
    assert select_revised(grid, {key: PASS_ALL for key in grid}) == (30, 0.96)  # 탐지가 더 빠르면 T 가 높아도 그쪽


def test_select_revised_filters_and_none_report_text():
    grid = {(30, 0.97): fake_cell(1, 3, 30),    # k=1: 오답 1건에 경고가 켜져 제외
            (50, 0.95): fake_cell(2, 5.5, 50),  # 회복 2초 간격 100초: 제외
            (30, 0.90): fake_cell(4, 15, 30),   # 원래 합격 실패: 제외
            (30, 0.95): fake_cell(2, 5.5, None)}  # 회복 없음: 제외
    verdicts = {key: PASS_ALL for key in grid}
    verdicts[(30, 0.90)] = {"all": False}
    assert select_revised(grid, verdicts) is None
    streams, v2 = _good()
    real = evaluate_grid(streams, v2, (10,), (0.95,))  # k=1 이라 후보 없음
    real_verdicts = {k: judge_cell(c, k[0]) for k, c in real.items()}
    judged = {42: {"v1": streams, "v2": v2}}
    report = build_report({42: real}, real_verdicts, (10, 0.95), judged, {"splits": {"train": {"size": 2000}}},
                          {"dark": 0.18}, (42,), 4, {"a": "1"}, {"a": "1"})
    assert "개정 규칙 선택 결과: 만족하는 (N, T) 조합 없음. 기준을 바꾸지 않고 멈춰서 보고한다." in report


def test_report_states_normal_wrong_count_and_flags_zero():
    streams, v2 = _good()
    grid = evaluate_grid(streams, v2, (10,), (0.95,))
    verdicts = {k: judge_cell(c, k[0]) for k, c in grid.items()}
    info, fp = {"splits": {"train": {"size": 2000}}}, {"a": "1"}
    wrong_normal = {**streams, "normal": [seq(("o", 290), ("x", 10))] * 4}
    judged = {42: {"v1": streams, "v2": v2}, 43: {"v1": wrong_normal, "v2": v2}}
    report = build_report({42: grid, 43: grid}, verdicts, (10, 0.95), judged, info, {"dark": 0.18}, (42, 43), 4, fp, fp)
    assert "시드 42 정상 스트림 오답 0건 / 1200건. 오답이 0건이라 헛경고 합격은 모든 (N, T) 칸에서 자동 통과이고 칸을 가르지 못한다" in report
    assert "참고(합격 판정에 쓰지 않음): 시드 43 정상 스트림 오답 40건 / 1200건" in report
    judged_wrong = {42: {"v1": wrong_normal, "v2": v2}}
    report = build_report({42: grid}, verdicts, (10, 0.95), judged_wrong, info, {"dark": 0.18}, (42,), 4, fp, fp)
    assert "시드 42 정상 스트림 오답 40건 / 1200건" in report and "칸을 가르지 못한다" not in report  # 오답이 있으면 단정하지 않는다


def test_revision_section_follows_original_rule_and_result():
    streams, v2 = _good()
    grid = evaluate_grid(streams, v2, (10,), (0.95,))
    verdicts = {k: judge_cell(c, k[0]) for k, c in grid.items()}
    judged = {42: {"v1": streams, "v2": v2}}
    report = build_report({42: grid}, verdicts, (10, 0.95), judged, {"splits": {"train": {"size": 2000}}},
                          {"dark": 0.18}, (42,), 4, {"a": "1"}, {"a": "1"})
    lines = report.splitlines()
    order = [lines.index(h) for h in ("== 선택 규칙 (측정 전에 고정) ==", "== 선택 결과 ==", "== 규칙 개정 (결과 확인 후) ==", "== 설정 ==")]
    assert order == sorted(order)
    assert lines[1] == f"조합 선택 규칙: {SELECTION_RULE}"  # 원래 규칙 줄은 그대로
    assert "선택된 조합: N=10, T=0.95" in report  # 원래 선택 결과도 그대로
    assert "개정 1·2는 모두 측정 결과를 확인한 뒤에 한 것이다." in report
    assert "개정 1:" in report and "개정 2:" in report and "필터로 먼저 적용한 뒤 N 최대, dark 탐지 중앙값 최소, T 최소 순" in report
