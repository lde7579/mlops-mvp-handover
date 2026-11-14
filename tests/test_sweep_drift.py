"""scripts/sweep_drift.py 테스트: 작은 단위 검사와, 아주 작은 건수로 끝까지 한 번 도는지 확인."""
import shutil

import numpy as np
import pytest

from parts.collector.config import load_config
from parts.collector.simulator import DRIFT_TYPES
from scripts.make_dataset import DATASET_DIR, DatasetError, PROJECT_ROOT
from scripts.sweep_drift import (
    BASELINE,
    DEFAULT_STRENGTHS,
    check_sweep_overlap,
    describe,
    false_alarm_rate,
    in_target,
    main,
    make_cell,
    parse_strengths,
    resolve_report_path,
    run_sweep,
)


def copy_dataset(template, dest):
    shutil.copytree(template / DATASET_DIR, dest / DATASET_DIR)  # 세션 템플릿(conftest 의 dataset_template)을 복사, 원본은 건드리지 않는다


def test_false_alarm_rate_is_fp_over_normals():
    assert false_alarm_rate([[140, 10], [5, 35]]) == pytest.approx(10 / 150)
    assert false_alarm_rate([[150, 0], [0, 40]]) == 0.0
    assert false_alarm_rate([[0, 0], [3, 7]]) == 0.0  # 정상이 없으면 0 으로 둔다


def test_target_band_boundaries_are_inclusive():
    assert in_target(0.70) and in_target(0.90) and in_target(133 / 190) and in_target(171 / 190)
    assert not in_target(0.699) and not in_target(0.9001)


@pytest.mark.parametrize("text", ["", "a", "0.1,0.1", "0", "-0.1", "1.01", "0.1,,0.2"])
def test_parse_strengths_rejects_bad_input(text):
    with pytest.raises(ValueError):
        parse_strengths(text)


def test_parse_strengths_accepts_list():
    assert parse_strengths("0.05,0.1,1") == (0.05, 0.1, 1.0)


def test_default_candidates_include_old_strengths_and_are_valid():
    # 기본 후보는 3-C 스윕용이라 옛 강도(0.30 / 0.11 / 0.04)를 포함한다. 강도는 3-D 에서 바뀌었다
    # (현재 collector.toml 값은 test_collector.py::test_default_config_values 가 확인한다).
    old = {"dark": 0.30, "noise": 0.11, "shift": 0.04}
    for kind, values in DEFAULT_STRENGTHS.items():
        assert old[kind] in values
        assert len(values) == 5 and parse_strengths(",".join(map(str, values)))


def test_describe_matches_simulator_formulas():
    assert describe("dark", 0.30) == "밝기 배율 0.730"
    assert describe("noise", 0.11) == "노이즈 표준편차 17.0"
    assert describe("shift", 0.04) == "위치 흔들림 ±3.00픽셀"


def test_cell_is_deterministic_and_valid():
    a = make_cell("dark", 0.1, n_normal=6, n_defect=2)
    b = make_cell("dark", 0.1, n_normal=6, n_defect=2)
    assert np.array_equal(a.images, b.images) and a.labels == b.labels and a.unfastened == b.unfastened
    assert a.images.dtype == np.uint8 and a.images.shape == (8, 128, 128)
    assert a.labels.count("NG") == 2 and a.labels.count("OK") == 6
    for label, parts in zip(a.labels, a.unfastened):
        if label == "OK":
            assert parts == []
        else:  # Collector 와 같은 규칙: 1~2개, 번호 1~6, 오름차순, 중복 없음
            assert 1 <= len(parts) <= 2 and parts == sorted(set(parts)) and all(1 <= p <= 6 for p in parts)
    assert not np.array_equal(a.images, make_cell("dark", 0.2, n_normal=6, n_defect=2).images)  # 강도가 다르면 다른 이미지
    assert not np.array_equal(a.images, make_cell("noise", 0.1, n_normal=6, n_defect=2).images)  # 종류가 다르면 다른 이미지


def test_overlap_with_dataset_or_between_cells_fails():
    check_sweep_overlap({"a": {"1"}, "b": {"2"}}, {"3"})
    with pytest.raises(DatasetError):
        check_sweep_overlap({"a": {"1"}, "b": {"2"}}, {"2", "3"})  # 데이터셋과 겹침
    with pytest.raises(DatasetError):
        check_sweep_overlap({"a": {"1"}, "b": {"1"}}, {"3"})  # 칸끼리 겹침


def test_sweep_runs_end_to_end_on_tiny_data(tmp_path, dataset_template):
    copy_dataset(dataset_template, tmp_path)
    config = tmp_path / "predictor.toml"
    config.write_text('threshold = 0.5\nepochs = 1\nbatch_size = 32\nlearning_rate = 0.001\n'
                      'seed = 42\nmodels_dir = "models"\n', encoding="utf-8")
    toml_path = PROJECT_ROOT / "parts" / "collector" / "collector.toml"
    toml_before = toml_path.read_bytes()
    current_dark = load_config().drift_strength["dark"]  # 현재 collector.toml 값 (강도를 바꿔도 이 테스트는 그대로)
    strengths = {"dark": (0.1, current_dark), "noise": (0.11,), "shift": (0.04,)}
    report_path = tmp_path / "data" / "sweep_report.txt"

    out = run_sweep(tmp_path, seeds=(42, 44), strengths=strengths, n_normal=6, n_defect=2,
                    predictor_config=config, report_path=report_path, log=lambda *_: None)

    text = report_path.read_text(encoding="utf-8")
    assert out["report"] == text.rstrip("\n")
    lines = text.splitlines()
    assert sum(line.startswith(f"{BASELINE}, 0.000, 시드") for line in lines) == 2
    for kind, values in strengths.items():
        for s in values:
            for seed in (42, 44):  # 시드별 줄
                assert any(line.startswith(f"{kind}, {s:.3f}") and f"시드 {seed}:" in line
                           and "오경보율" in line and "목표구간" in line for line in lines)
            assert any(line.startswith(f"{kind}, {s:.3f}") and ", 평균:" in line for line in lines)  # 평균 줄
    for kind in DRIFT_TYPES:
        assert any(line.startswith(f"{kind}, 평균 정확도가 목표 구간 안인 강도:") for line in lines)
    # 현재 collector.toml 값인 dark 칸에만 "현재값" 표시가 붙는다 (시드 2개 줄 + 평균 줄 = 3줄)
    assert f"(현재값, {describe('dark', current_dark)})" in text
    assert sum("(현재값, " in line for line in lines) == 3
    assert not (tmp_path / "models").exists()  # 모델은 임시 폴더에만 생기고 지워진다
    assert toml_path.read_bytes() == toml_before  # collector.toml 은 그대로


# ---------- 명령줄 옵션 (--n-normal, --n-defect, --report) ----------

def test_cli_options_set_counts_and_report_path(tmp_path, capsys, dataset_template):
    copy_dataset(dataset_template, tmp_path)
    target = tmp_path / "data" / "sweep_report_narrow.txt"

    code = main(["--seeds", "42", "--dark", "0.2", "--noise", "0.2", "--shift", "0.1",
                 "--n-normal", "6", "--n-defect", "2", "--report", "data/sweep_report_narrow.txt"], root=tmp_path)

    assert code == 0
    text = target.read_text(encoding="utf-8")
    assert "종류·강도마다 정상 6장 + 불량 2장" in text
    assert not (tmp_path / "data" / "sweep_report.txt").exists()  # 기본 위치에는 쓰지 않는다
    assert "sweep_report_narrow.txt" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["--n-normal", "0"], ["--n-defect", "-1"]])
def test_cli_rejects_bad_counts(tmp_path, argv):
    assert main(argv, root=tmp_path) == 1  # 데이터셋을 읽기 전에 거부한다


def test_cli_rejects_non_integer_counts(tmp_path):
    with pytest.raises(SystemExit) as e:
        main(["--n-normal", "abc"], root=tmp_path)
    assert e.value.code == 2  # argparse 오류


def test_report_outside_project_root_is_rejected(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    for text in (str(outside), "../outside.txt", str(root / ".." / "outside.txt")):
        assert main(["--report", text], root=root) == 1
        assert not outside.exists()
    inside = resolve_report_path(root, "data/x.txt")  # 루트 안은 허용
    assert inside == (root / "data" / "x.txt").resolve()
    assert resolve_report_path(root, str(root / "y.txt")) == (root / "y.txt").resolve()
