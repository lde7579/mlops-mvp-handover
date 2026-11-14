"""parts/collector 보조 테스트 (계약 테스트는 tests/contract/ 에 있다)."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from common.schemas import Label, SchemaError
from parts.collector import Collector
from parts.collector import collector as collector_module
from parts.collector.config import ConfigError, load_config
from parts.collector.simulator import BOLT_COUNT, CENTERS, DRIFT_TYPES

FIXED_TIME = datetime(2026, 10, 18, 12, 0, 0, tzinfo=timezone(timedelta(hours=9)))
SPLITS = ["train", "pool", "test_normal", "test_drift"]


@pytest.fixture
def collector(tmp_path):
    return Collector(root=tmp_path, clock=lambda: FIXED_TIME)


@pytest.fixture
def no_png(monkeypatch):
    """이미지를 파일로 저장하지 않는다 (통계 테스트 6개에서만 쓴다).

    이 테스트들은 next_sample().image(메모리의 이미지)만 보고 파일은 보지 않는다. PNG 저장과 다시 열어 검사하는 단계가
    이미지 1장 시간의 약 58% 라서 생략하면 같은 이미지(난수는 저장과 무관해서 100장이 한 장도 다르지 않음, PNG 는 무손실)로
    같은 기준을 훨씬 빨리 확인한다. 저장·재검사는 이 파일의 test_image_path_is_relative_and_file_exists,
    test_failed_check_leaves_no_file_and_keeps_id 와 tests/contract/test_collector_contract.py 가 확인한다.
    수집기의 내부 메서드 _save_png 에 의존하므로, 수집기가 바뀌어 이 이름이 없어지면 여기서 바로 실패한다.
    """
    monkeypatch.setattr(Collector, "_save_png", lambda self, inspection_id, path, image: None)


def center_means(image):
    """볼트 6개 각각의 중심에서 반지름 5 이내의 평균 밝기."""
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    return [float(image[(xx - cx) ** 2 + (yy - cy) ** 2 <= 25].mean()) for cx, cy in CENTERS]


# ---------- 설정 ----------

def test_default_config_values():
    cfg = load_config()
    assert cfg.defect_rate == 0.2
    assert cfg.image_size == 128 and cfg.part_count == 6
    assert cfg.seed == 42 and cfg.data_dir == "data"
    assert cfg.drift_strength == {"dark": 0.18, "noise": 0.24, "shift": 0.11}  # 3-D 에서 확정 (data/sweep_report_narrow.txt)


BASE_TOML = """defect_rate = 0.2
image_size = 128
part_count = 6
seed = 42
data_dir = "data"
[drift_strength]
dark = 0.30
noise = 0.11
shift = 0.04
"""


@pytest.mark.parametrize(
    "old,new",
    [
        ("part_count = 6", "part_count = 7"),
        ("image_size = 128", "image_size = 64"),
        ("defect_rate = 0.2", "defect_rate = 1.5"),
        ("defect_rate = 0.2", 'defect_rate = "0.2"'),
        ("seed = 42", "seed = 4.2"),
        ('data_dir = "data"', 'data_dir = "C:\\\\data"'),
        ('data_dir = "data"', 'data_dir = "/data"'),
        ("shift = 0.04", ""),
        ("dark = 0.30", "dark = 2.0"),
        ("seed = 42", "seed = 42\nunknown = 1"),
    ],
)
def test_config_rejects_bad_values(tmp_path, old, new):
    path = tmp_path / "bad.toml"
    path.write_text(BASE_TOML.replace(old, new), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_config_base_toml_is_valid(tmp_path):
    path = tmp_path / "ok.toml"
    path.write_text(BASE_TOML, encoding="utf-8")
    assert load_config(path).seed == 42


# ---------- ID와 저장 ----------

def test_ids_unique_sequential_across_splits(collector):
    ids = [collector.next_sample(data_split=SPLITS[i % 4]).inspection.inspection_id for i in range(8)]
    assert ids == [f"INS-{n:06d}" for n in range(1, 9)]


def test_ids_continue_after_restart(tmp_path):
    first = Collector(root=tmp_path)
    for _ in range(3):
        first.next_sample()
    second = Collector(root=tmp_path)
    assert second.next_sample().inspection.inspection_id == "INS-000004"


def test_samples_folder_is_ignored_for_ids(tmp_path):
    (tmp_path / "data" / "samples").mkdir(parents=True)
    (tmp_path / "data" / "samples" / "INS-000099.png").write_bytes(b"x")
    assert Collector(root=tmp_path).next_sample().inspection.inspection_id == "INS-000001"


def test_existing_file_is_never_overwritten(tmp_path):
    c = Collector(root=tmp_path)
    taken = tmp_path / "data" / "pool" / "INS-000001.png"
    taken.parent.mkdir(parents=True)
    taken.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        c.next_sample()
    assert taken.read_bytes() == b"keep"


def test_image_path_is_relative_and_file_exists(collector, tmp_path):
    for split in SPLITS:
        s = collector.next_sample(data_split=split)
        path = s.inspection.image_path
        assert path == f"data/{split}/{s.inspection.inspection_id}.png"
        assert (tmp_path / path).is_file()


def test_failed_check_leaves_no_file_and_keeps_id(tmp_path, monkeypatch):
    c = Collector(root=tmp_path)

    def boom(_):
        raise SchemaError("강제 실패")

    monkeypatch.setattr(collector_module, "validate_pil_image", boom)
    with pytest.raises(SchemaError):
        c.next_sample()
    assert list((tmp_path / "data").rglob("*.png")) == []

    monkeypatch.undo()
    assert c.next_sample().inspection.inspection_id == "INS-000001"


# ---------- 볼트 체결 유무 ----------

def test_ng_has_one_or_two_unfastened_bolts_ok_has_none(collector, no_png):
    seen = set()
    for _ in range(60):
        ng = collector.next_sample(label="NG").inspection
        assert ng.true_label is Label.NG
        parts = ng.missing_parts
        assert 1 <= len(parts) <= 2 and parts == sorted(set(parts))
        assert all(1 <= p <= BOLT_COUNT for p in parts)
        seen.update(parts)
        assert collector.next_sample(label="OK").inspection.missing_parts == []
    assert seen == set(range(1, BOLT_COUNT + 1))


def test_fastened_bolt_center_is_bright_and_hole_center_is_dark(collector, no_png):
    # 기준 85: 정상 환경 400장(체결 1,800개, 미체결 600개)을 실측한 값은
    # 체결 최소 140.9(평균 172.8), 미체결 최대 28.1(평균 21.2). 두 값의 가운데(약 84.5)에서 양쪽 약 56씩 여유.
    for _ in range(30):
        s = collector.next_sample(label="NG")
        for number, mean in enumerate(center_means(s.image), start=1):
            if number in s.inspection.missing_parts:
                assert mean < 85, f"{number}번 볼트 구멍 자리가 너무 밝음: {mean:.0f}"
            else:
                assert mean > 85, f"{number}번 볼트 윗면 자리가 너무 어두움: {mean:.0f}"


# ---------- 시나리오·드리프트 ----------

def test_normal_scenario_has_no_drift_type(collector):
    s = collector.next_sample().inspection
    assert s.scenario.value == "normal" and s.drift_type is None


def test_drift_scenario_picks_all_three_types(collector, no_png):
    types = {collector.next_sample(scenario="drift").inspection.drift_type for _ in range(60)}
    assert types == set(DRIFT_TYPES)


def test_drift_type_can_be_chosen(collector):
    s = collector.next_sample(scenario="drift", drift_type="dark").inspection
    assert s.drift_type == "dark"


def test_dark_drift_is_darker(tmp_path, no_png):
    c = Collector(root=tmp_path)
    normal = np.mean([c.next_sample(label="OK").image.mean() for _ in range(30)])
    dark = np.mean([c.next_sample(scenario="drift", drift_type="dark", label="OK").image.mean() for _ in range(30)])
    # 기준 0.95 의 의미: "dark 이미지가 정상보다 평균 5% 이상 어둡다".
    # 근거 (dark 강도 0.18): 이론 밝기 비율 = 1 - 0.9 x 0.18 = 0.838. 이 테스트의 측정값은 0.878.
    # 이미지마다 밝기가 무작위(x0.85~1.15)라서 30장 평균의 비율에는 우연 오차(표준편차 약 0.022)가 있고,
    # 측정값이 이론보다 0.04 높게 나왔다. 옛 기준 0.9 는 측정값에서 약 1 표준편차밖에 여유가 없었다.
    # 0.95 는 측정값에서 약 3 표준편차, 이론값에서 약 5 표준편차 위다 (옛 강도 0.30: 이론 0.730, 측정 0.764).
    assert dark < normal * 0.95


def test_noise_drift_is_noisier(tmp_path, no_png):
    c = Collector(root=tmp_path)

    # 볼트가 없는 위쪽 여백(0~12행)의 표준편차. 실측(30장 평균, noise 강도 0.24): 정상 6.19, noise 29.83 (약 4.8배).
    # (옛 강도 0.11 에서는 정상 6.19, noise 16.97, 약 2.7배. 정상 100장 실측 범위는 5.9~6.4.)
    # 19행까지 넓히면 플랜지 윗부분이 들어와 정상이 최대 12.2까지 커지므로 12행까지만 본다.
    def margin_std(**kw):
        return float(np.mean([c.next_sample(label="OK", **kw).image[:13].std() for _ in range(30)]))

    assert margin_std(scenario="drift", drift_type="noise") > margin_std() * 1.5


def test_shift_drift_moves_bolts_more(tmp_path, no_png):
    c = Collector(root=tmp_path)

    # 밝은 부분(볼트 윗면)의 가로 중심이 흔들리는 정도. 밝기 변화에 덜 민감하도록
    # 기준을 이미지 자체의 중앙값(판 밝기) x 1.6 으로 둔다. 실측(60장, shift 강도 0.11): 정상 0.37, shift 3.06 (약 8.3배).
    # (옛 강도 0.04 에서는 정상 0.37, shift 1.95, 약 5.3배.)
    def centroid_x_std(**kw):
        xs = []
        for _ in range(60):
            img = c.next_sample(label="OK", **kw).image.astype(float)
            xs.append(np.nonzero(img > np.median(img) * 1.6)[1].mean())
        return float(np.std(xs))

    assert centroid_x_std(scenario="drift", drift_type="shift") > centroid_x_std() * 1.5


# ---------- 잘못된 인자 ----------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"scenario": "other"},
        {"data_split": "validation"},
        {"label": "ng"},
        {"drift_type": "dark"},  # normal 인데 drift_type
        {"scenario": "drift", "drift_type": "blur"},
    ],
)
def test_next_sample_rejects_bad_arguments(collector, kwargs):
    with pytest.raises(ValueError):
        collector.next_sample(**kwargs)


def test_rejected_arguments_do_not_use_an_id(collector):
    with pytest.raises(ValueError):
        collector.next_sample(scenario="other")
    assert collector.next_sample().inspection.inspection_id == "INS-000001"


def test_seed_must_be_int(tmp_path):
    with pytest.raises(ValueError):
        Collector(root=tmp_path, seed="42")
