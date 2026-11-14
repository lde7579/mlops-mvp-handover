"""scripts/measure_latency.py: 서버를 띄우지 않는 단위 시험 (통계, 판정 경계, 임시 루트 보호, 인자·경로 검증, 보고서 형식, 항목 5 감싸기).

실제 측정(서버 기동)은 python -m scripts.measure_latency 로 직접 실행해서 확인한다. 이 시험은 torch·backend 를 가져오지 않는다
(마지막의 '제품 내부 이름' 시험만 backend 가 있을 때 돌고, 없으면 건너뛴다).
"""
import importlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import measure_latency as m


# ---------- 통계 ----------

def test_percentile_is_nearest_rank():
    values = list(range(1, 101))
    assert (m.percentile(values, 50), m.percentile(values, 95), m.percentile(values, 99), m.percentile(values, 100)) == (50, 95, 99, 100)
    assert m.percentile(values, 0) == 1  # 하한: 최솟값


def test_percentile_edge_cases():
    assert m.percentile([7], 99) == 7
    assert m.percentile([9, 1, 5], 100) == 9 and m.percentile([9, 1, 5], 50) == 5  # 정렬되지 않은 입력
    assert m.percentile([1, 2, 3, 4, 5, 6, 7], 95) == 7  # 올림: 7 * 0.95 = 6.65 -> 7번째
    with pytest.raises(ValueError):
        m.percentile([], 95)


def test_summarize():
    s = m.summarize([10, 20, 30, 40, 1000])
    assert (s["n"], s["min"], s["max"], s["mean"], s["median"]) == (5, 10, 1000, 220, 30)
    assert (s["p95"], s["p99"]) == (1000, 1000) and s["margin_pct"] == pytest.approx(80.0)
    assert m.summarize([]) is None
    one = m.summarize([3.5])
    assert one["min"] == one["max"] == one["median"] == 3.5


@pytest.mark.parametrize("max_ms, margin", [(0, 100.0), (1000, 80.0), (2500, 50.0), (5000, 0.0), (6000, -20.0)])
def test_margin_against_5000ms(max_ms, margin):
    assert m.margin_pct(max_ms) == pytest.approx(margin)


@pytest.mark.parametrize("max_ms, passed", [(0, True), (4999.9, True), (5000.0, True), (5000.1, False), (60000, False)])
def test_verdict_boundary_exactly_5000_passes(max_ms, passed):
    assert m.verdict(max_ms) is passed


def test_limit_is_the_documented_5_seconds():
    assert m.LIMIT_MS == 5000.0


def test_fmt():
    assert "여유 80.0%" in m.fmt(m.summarize([1000])) and "최댓값 1000.0ms" in m.fmt(m.summarize([1000]))
    assert m.fmt(None) == "(자료 없음)"
    assert "여유 -20.0%" in m.fmt(m.summarize([6000]))


# ---------- 인자 ----------

def parse(*argv):
    return m.settings_from_args(m.build_parser().parse_args(list(argv)))


def test_full_defaults():
    s = parse()
    assert (s["restarts"], s["warm"], s["pairs"], s["lazy"], s["idle_seconds"], s["quick"]) == (5, 300, 200, 3, 30, False)
    assert s["levels"] == ((4, 75), (2, 100), (8, 25))


def test_quick_defaults_are_small_and_skip_the_idle_wait():
    s = parse("--quick")
    assert (s["restarts"], s["warm"], s["pairs"], s["lazy"], s["idle_seconds"], s["quick"]) == (1, 20, 10, 1, 0, True)
    assert sum(w * each for w, each in s["levels"]) < 60


def test_explicit_options_win_over_quick():
    s = parse("--quick", "--warm", "7", "--idle-seconds", "2")
    assert s["warm"] == 7 and s["idle_seconds"] == 2 and s["restarts"] == 1


def test_idle_seconds_zero_is_allowed():
    assert parse("--idle-seconds", "0")["idle_seconds"] == 0


@pytest.mark.parametrize("argv", [["--warm", "0"], ["--restarts", "-1"], ["--pairs", "abc"], ["--lazy", "0"], ["--idle-seconds", "-1"],
                                  ["--warm", "2.5"], ["--restarts", ""]])
def test_bad_numbers_are_rejected_with_exit_code_2(argv):
    with pytest.raises(SystemExit) as e:
        m.build_parser().parse_args(argv)
    assert e.value.code == 2


# ---------- 임시 루트 보호 ----------

def test_temp_root_outside_the_project_is_accepted(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    other = tmp_path / "temp_root"
    other.mkdir()
    assert m.assert_safe_root(other, project) == other.resolve()


@pytest.mark.parametrize("make", [
    lambda project: project,                       # 프로젝트 자체
    lambda project: project / "data",              # 실제 data/
    lambda project: project / "models" / "v1",     # 실제 models/
    lambda project: project / "x" / "..",          # '..' 로 프로젝트로 돌아옴
    lambda project: project.parent,                # 프로젝트를 품은 폴더
])
def test_temp_root_overlapping_the_project_is_refused(tmp_path, make):
    project = tmp_path / "project"
    (project / "x").mkdir(parents=True)
    with pytest.raises(ValueError, match="겹칩니다"):
        m.assert_safe_root(make(project), project)


def test_the_default_project_root_is_this_repository():
    assert (m.PROJECT_ROOT / "scripts" / "measure_latency.py").is_file()
    with pytest.raises(ValueError):
        m.assert_safe_root(m.PROJECT_ROOT / "data")


# ---------- 보고서 경로 ----------

def test_report_path_rules(tmp_path):
    assert m.resolve_report_path("-", tmp_path) is None
    assert m.resolve_report_path("data/latency_report.txt", tmp_path) == (tmp_path / "data" / "latency_report.txt").resolve()
    assert m.resolve_report_path(str(tmp_path / "data" / "x.txt"), tmp_path) == (tmp_path / "data" / "x.txt").resolve()
    for outside in ("../outside.txt", str(tmp_path.parent / "outside.txt"), "data/../../outside.txt"):
        with pytest.raises(ValueError, match="프로젝트 루트"):
            m.resolve_report_path(outside, tmp_path)


def test_existing_report_is_never_overwritten_without_the_flag(tmp_path):
    target = tmp_path / "report.txt"
    m.check_report_target(target, overwrite=False)  # 없으면 괜찮다
    target.write_text("지우면 안 되는 보고서", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--overwrite"):
        m.check_report_target(target, overwrite=False)
    m.check_report_target(target, overwrite=True)
    m.check_report_target(None, overwrite=False)  # '-' 는 파일이 없다
    assert target.read_text(encoding="utf-8") == "지우면 안 되는 보고서"


# ---------- 준비물 확인 ----------

def make_project(root, model=True, collector=True):
    if model:
        (root / "models" / "v1").mkdir(parents=True)
        (root / "models" / "v1" / "model.pt").write_bytes(b"weights")
        (root / "models" / "v1" / "metrics.json").write_text("{}", encoding="utf-8")
    if collector:
        (root / "parts" / "collector").mkdir(parents=True)
        (root / "parts" / "collector" / "collector.toml").write_text("seed = 1\n", encoding="utf-8")
    return root


def test_prerequisites(tmp_path):
    assert m.check_prerequisites(make_project(tmp_path / "ok")) == []
    empty = m.check_prerequisites(tmp_path / "none")
    assert len(empty) == 3 and any("train_v1" in p for p in empty) and any("collector.toml" in p for p in empty)
    assert len(m.check_prerequisites(make_project(tmp_path / "nomodel", model=False))) == 2
    assert len(m.check_prerequisites(make_project(tmp_path / "nocollector", collector=False))) == 1


# ---------- 실제 파일 전후 비교 ----------

def test_real_files_snapshot_detects_changes(tmp_path):
    make_project(tmp_path)
    (tmp_path / "data" / "pool").mkdir(parents=True)
    (tmp_path / "data" / "mlops.db").write_bytes(b"db")
    (tmp_path / "data" / "latency_report.txt").write_text("보고서", encoding="utf-8")
    before = m.real_files(tmp_path)
    assert m.real_files(tmp_path) == before and before["data/pool 파일 수"] == 0 and before["mlops.db"][0] == 2
    assert len(before["models/v1/model.pt"]) == 64 and before["mlops.db-wal"] is None
    (tmp_path / "models" / "v1" / "model.pt").write_bytes(b"changed")
    assert m.real_files(tmp_path) != before  # 모델 파일 해시가 달라진다
    (tmp_path / "models" / "v1" / "model.pt").write_bytes(b"weights")
    (tmp_path / "data" / "pool" / "a.png").write_bytes(b"x")
    assert m.real_files(tmp_path)["data/pool 파일 수"] == 1
    (tmp_path / "data" / "latency_report.txt").write_text("다른 보고서", encoding="utf-8")
    assert m.real_files(tmp_path)[m.DEFAULT_REPORT] != before[m.DEFAULT_REPORT]  # 기존 보고서가 바뀌면 드러난다


# ---------- 항목 5: 함수 감싸기 ----------

def fake_product():
    class Repo:
        def get(self, x):
            return x * 2

    def helper(x):
        if x < 0:
            raise ValueError("음수")
        return x + 1

    module = SimpleNamespace(Repo=Repo, helper=helper)
    targets = (("fake", "Repo", "get", "repo_get"), ("fake", None, "helper", "helper"))
    return module, targets, (lambda path: module)


def test_wrapped_functions_return_the_same_values_and_record_times():
    module, targets, resolve = fake_product()
    original_get, original_helper = module.Repo.get, module.helper
    restores = m.install_timers(targets, resolve)
    d = {}
    token = m.CURRENT.set(d)
    try:
        assert module.Repo().get(4) == 8 and module.helper(1) == 2
        with pytest.raises(ValueError):
            module.helper(-1)  # 예외도 그대로 나가고 시간은 남는다
    finally:
        m.CURRENT.reset(token)
    assert len(d["repo_get"]) == 1 and len(d["helper"]) == 2 and all(v >= 0 for v in d["helper"])
    m.restore_timers(restores)
    assert module.Repo.get is original_get and module.helper is original_helper


def test_nothing_is_recorded_outside_a_request():
    module, targets, resolve = fake_product()
    restores = m.install_timers(targets, resolve)
    assert module.helper(1) == 2 and m.CURRENT.get() is None
    m.restore_timers(restores)


def test_missing_product_name_changes_nothing_and_names_what_is_missing():
    module, targets, resolve = fake_product()
    original = module.helper
    broken = targets + (("fake", None, "was_renamed", "x"), ("fake", "Gone", "get", "y"))
    with pytest.raises(m.TimerError) as e:
        m.install_timers(broken, resolve)
    assert "was_renamed" in str(e.value) and "Gone" in str(e.value)
    assert module.helper is original  # 일부만 감싸진 상태가 남지 않는다


def test_missing_module_is_also_a_timer_error():
    def resolve(path):
        raise ImportError(path)

    with pytest.raises(m.TimerError):
        m.install_timers((("no.such.module", None, "f", "k"),), resolve)


def test_timer_targets_are_well_formed():
    assert len(m.TIMER_TARGETS) == 8
    assert all(len(t) == 4 for t in m.TIMER_TARGETS) and len({t[3] for t in m.TIMER_TARGETS}) == 8


def test_timer_targets_still_exist_in_the_product():
    pytest.importorskip("backend.main")  # 조원 전달 압축에 backend/ 가 없는 동안은 건너뛴다
    classes = [(importlib.import_module(mod), owner, attr) for mod, owner, attr, _ in m.TIMER_TARGETS]
    before = [getattr(getattr(module, owner) if owner else module, attr) for module, owner, attr in classes]
    restores = m.install_timers(m.TIMER_TARGETS, importlib.import_module)  # 제품 내부 이름이 바뀌면 여기서 깨진다
    m.restore_timers(restores)
    after = [getattr(getattr(module, owner) if owner else module, attr) for module, owner, attr in classes]
    assert all(a is b for a, b in zip(before, after)), "감싼 함수를 원래대로 되돌리지 못했습니다"


# ---------- 보고서 ----------

def fake_result(max_b=20.0, skip_breakdown=False, changed=False, with_idle=True):
    row = {"t_up_ms": 5000.0, "t_loaded_ms": 5100.0, "cold_B": 40.0, "cold_A": 5, "cold_status": 201, "second_B": 12.0, "second_status": 201}
    if with_idle:
        row.update(idle_B=30.0, idle_A=2, idle_status=201)
    warm = [{"status": 201, "B": 9.0 + i * 0.01, "A": 2} for i in range(20)]
    pairs = [{"gen_status": 201, "gen_ms": 30.0, "pred_status": 201, "B": 9.5, "A": 2, "pair_ms": 40.0} for _ in range(5)]
    conc = {str(w): {"rows": [{"status": 201, "B": max_b, "A": 3}] * 4, "wall_s": 0.5, "rps": 8.0} for w in (2, 4, 8)}
    comp_names = ("auth", "get_inspection", "resolve_image", "read_image", "commit", "server_internal")
    breakdown = ({"skipped": "제품 내부 이름을 찾지 못했습니다: backend.x.y"} if skip_breakdown else
                 {"startup_ms": 3000.0, "n_records": 5, "B": [9.0] * 5, "A": [2] * 5, "statuses": [201],
                  "components": {k: [1.0, 1.2, 0.9, 1.1, 1.0] for k in comp_names}})
    state = {"time": "2026-10-18T10:00:00", "cpu_pct": "10.5", "python_processes": "2", "top_memory": ["chrome:100MB"]}
    real = {"mlops.db": [100, 1], "models/v1/model.pt": "ab" * 32}
    return {"started": "2026-10-18T10:00:00", "finished": "2026-10-18T10:05:00", "quick": False, "pc_before": state, "pc_after": state,
            "startup_cold_idle": [row, dict(row)], "warm_pairs_concurrent": {"t_up_ms": 5000.0, "t_loaded_ms": 5100.0, "first": {"B": 40.0},
            "warm": warm, "pairs": pairs, "concurrent": conc},
            "lazy": [{"t_up_ms": 1500.0, "before_status": 503, "first_B": 3800.0, "first_A": 5, "first_status": 201, "second_B": 11.0, "second_status": 201}],
            "breakdown": breakdown, "servers": [{"port": 5000, "pid": 1, "port_closed": True}],
            "real_before": real, "real_after": dict(real, **({"mlops.db": [101, 2]} if changed else {}))}


def test_report_has_every_section_and_passes_when_under_5000():
    text = m.build_report(fake_result(max_b=20.0))
    for section in ("[항목 1]", "[항목 2-콜드]", "[추가 a]", "[항목 2-웜]", "[항목 3]", "[항목 4 + 추가 b]", "2개 병렬", "4개 병렬", "8개 병렬",
                    "[참고] 지연 로딩", "[항목 5]", "종합 판정", "서버 종료 확인", "PC 상태 측정 전", "PC 상태 측정 후"):
        assert section in text, section
    assert "-> 통과" in text and "초과" not in text and "동일" in text and "닫힘=True" in text


def test_report_flags_a_value_over_5000ms():
    text = m.build_report(fake_result(max_b=5000.1))
    assert "5000ms 초과" in text and "-> 통과" not in text


def test_report_passes_at_exactly_5000ms():
    assert "-> 통과" in m.build_report(fake_result(max_b=5000.0))


def test_report_when_the_breakdown_was_skipped_still_judges_items_1_to_4():
    text = m.build_report(fake_result(skip_breakdown=True))
    assert "[항목 5] 구간 분해: 건너뜀" in text and "TIMER_TARGETS" in text and "종합 판정" in text and "-> 통과" in text


def test_report_flags_changed_real_files():
    assert "** 달라짐 **" in m.build_report(fake_result(changed=True))


def test_report_without_the_idle_item_omits_it():
    assert "[추가 a]" not in m.build_report(fake_result(with_idle=False))


# ---------- main: 측정을 시작하기 전에 거부하는 경우 ----------

def test_main_refuses_an_existing_report_before_measuring(tmp_path, capsys):
    make_project(tmp_path)
    (tmp_path / "data").mkdir()
    keep = tmp_path / "data" / "latency_report.txt"
    keep.write_text("기존 보고서", encoding="utf-8")
    assert m.main([], project_root=tmp_path) == 1
    assert "--overwrite" in capsys.readouterr().err and keep.read_text(encoding="utf-8") == "기존 보고서"


def test_main_refuses_a_report_outside_the_project(tmp_path, capsys):
    assert m.main(["--report", "../outside.txt"], project_root=tmp_path) == 1
    assert "프로젝트 루트" in capsys.readouterr().err


def test_main_says_how_to_prepare_when_the_model_is_missing(tmp_path, capsys):
    assert m.main([], project_root=tmp_path) == 1
    assert "train_v1" in capsys.readouterr().err


# ---------- 가져오기만으로는 무거운 것이 올라오지 않는다 ----------

def test_importing_the_script_does_not_load_torch_or_the_backend():
    code = "import sys, scripts.measure_latency; print(any(n in sys.modules for n in ('torch', 'backend', 'uvicorn', 'parts')))"
    out = subprocess.run([sys.executable, "-c", code], cwd=m.PROJECT_ROOT, capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "False", out.stderr
