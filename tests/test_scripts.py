"""scripts/ 테스트: make_dataset(생성·겹침 검사·재현성·삭제 보호)과 measure_baseline(작은 데이터로 끝까지 한 번).

모두 작은 건수와 임시 폴더만 쓴다. 실제 data/ 와 models/ 는 건드리지 않는다.
"""
import hashlib
import json
import shutil

import pytest

from common.schemas import ClassMetrics
from scripts.make_dataset import (
    DATASET_DIR,
    DEFAULT_SPECS,
    INFO,
    MANIFEST,
    DatasetError,
    RefusedPathError,
    SplitSpec,
    build_dataset,
    check_no_overlap,
    check_unique_ids,
    load_dataset,
    remove_dataset,
)
from scripts.measure_baseline import gates, parse_seeds, run_baseline

SPLITS = ["train", "pool", "test_normal", "test_drift"]


def specs(train=10, pool=5, normal=5, drift=6, seeds=(1, 2, 3, 4)):
    return (
        SplitSpec("train", train, "normal", seeds[0]),
        SplitSpec("pool", pool, "drift", seeds[1]),
        SplitSpec("test_normal", normal, "normal", seeds[2]),
        SplitSpec("test_drift", drift, "drift", seeds[3]),
    )


def read_rows(root):
    lines = (root / DATASET_DIR / MANIFEST).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


# 세트별 크기. 불량 정확히 20%(5의 배수)와 드리프트 종류 균등 분배(3의 배수)를 pool·test_drift 에서 확인하려면 15장이 최소다.
# train·test_normal 은 드리프트가 없어 5의 배수이기만 하면 되므로 작게 둔다 (만드는 시간을 줄이려는 것).
SMALL = {"train": 10, "pool": 15, "test_normal": 5, "test_drift": 15}
SMALL_TOTAL = sum(SMALL.values())


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    root = tmp_path_factory.mktemp("dataset")
    result = build_dataset(root, specs(SMALL["train"], SMALL["pool"], SMALL["test_normal"], SMALL["test_drift"]), log=lambda *_: None)
    return root, result


def copy_dataset(template, dest):
    shutil.copytree(template / DATASET_DIR, dest / DATASET_DIR)


# ---------- 기본값이 문서와 맞는가 ----------

def test_default_specs_follow_documents():
    sizes = {s.split: s.size for s in DEFAULT_SPECS}
    assert sizes["test_normal"] == 500 and sizes["test_drift"] == 500  # 문서 값
    assert sizes["train"] == 2000 and sizes["pool"] == 500  # 가정 (스파이크 건수)
    assert {s.scenario for s in DEFAULT_SPECS if s.split in ("train", "test_normal")} == {"normal"}
    assert {s.scenario for s in DEFAULT_SPECS if s.split in ("pool", "test_drift")} == {"drift"}
    assert len({s.seed for s in DEFAULT_SPECS}) == 4  # 세트마다 시드가 다르다


# ---------- make_dataset: 생성 ----------

def test_counts_defect_ratio_and_scenarios(small):
    root, result = small
    assert result["status"] == "created"
    rows = read_rows(root)
    assert len(rows) == SMALL_TOTAL
    scenarios = {"train": "normal", "pool": "drift", "test_normal": "normal", "test_drift": "drift"}
    expected = {split: (size, size // 5, scenarios[split]) for split, size in SMALL.items()}  # 불량은 정확히 20%
    for split, (size, ng, scenario) in expected.items():
        part = [r for r in rows if r["data_split"] == split]
        assert len(part) == size
        assert sum(r["true_label"] == "NG" for r in part) == ng  # 불량 정확히 20%
        assert {r["scenario"] for r in part} == {scenario}
        assert result["info"]["splits"][split]["ng"] == ng


def test_drift_types_are_even_and_recorded(small):
    root, _ = small
    rows = read_rows(root)
    for split in ("pool", "test_drift"):
        kinds = [r["drift_type"] for r in rows if r["data_split"] == split]
        assert sorted(set(kinds)) == ["dark", "noise", "shift"]
        assert {k: kinds.count(k) for k in set(kinds)} == {"dark": 5, "noise": 5, "shift": 5}
    for split in ("train", "test_normal"):
        assert {r["drift_type"] for r in rows if r["data_split"] == split} == {None}


def test_images_are_saved_under_dataset_dir_with_hashes(small):
    root, _ = small
    for r in read_rows(root):
        assert r["image_path"] == f"{DATASET_DIR}/{r['data_split']}/{r['inspection_id']}.png"
        assert hashlib.md5((root / r["image_path"]).read_bytes()).hexdigest()  # 파일이 있다
    data, info = load_dataset(root)  # 읽을 때 해시와 규칙을 다시 확인한다
    assert {s: len(d.labels) for s, d in data.items()} == SMALL
    assert info["total"] == SMALL_TOTAL


def test_inspection_ids_are_unique_across_all_four_sets(small):
    root, _ = small
    ids = [r["inspection_id"] for r in read_rows(root)]
    assert len(ids) == SMALL_TOTAL and len(set(ids)) == SMALL_TOTAL
    check_unique_ids(read_rows(root))
    pngs = sorted(p.name for p in (root / DATASET_DIR).rglob("*.png"))
    assert len(pngs) == len(set(pngs)) == SMALL_TOTAL  # 폴더가 달라도 파일 이름(ID)이 겹치지 않는다


def test_no_image_is_shared_between_sets(small):
    root, result = small
    rows = read_rows(root)
    hashes = {s: {r["image_md5"] for r in rows if r["data_split"] == s} for s in SPLITS}
    check_no_overlap(hashes)
    assert result["info"]["image_overlap_between_splits"] == 0


# ---------- make_dataset: 검사 함수 ----------

def test_check_no_overlap_detects_shared_image():
    check_no_overlap({"a": {"1", "2"}, "b": {"3"}})
    with pytest.raises(DatasetError, match="같은 이미지"):
        check_no_overlap({"a": {"1", "2"}, "b": {"2", "3"}})


def test_check_unique_ids_detects_duplicate():
    with pytest.raises(DatasetError, match="겹칩니다"):
        check_unique_ids([{"inspection_id": "INS-000001"}, {"inspection_id": "INS-000001"}])


def test_build_fails_and_cleans_up_when_sets_overlap(tmp_path):
    # train 과 test_normal 에 같은 시드를 주면 이미지가 똑같아져서 겹친다
    same = (SplitSpec("train", 5, "normal", 5), SplitSpec("pool", 5, "drift", 2),
            SplitSpec("test_normal", 5, "normal", 5), SplitSpec("test_drift", 5, "drift", 3))
    with pytest.raises(DatasetError, match="같은 이미지"):
        build_dataset(tmp_path, same, log=lambda *_: None)
    assert not (tmp_path / DATASET_DIR).exists()  # 반쪽짜리 데이터셋을 남기지 않는다


@pytest.mark.parametrize("bad", [
    (SplitSpec("train", 5, "normal", 1),) * 2,  # 같은 세트 두 번
    (SplitSpec("validation", 5, "normal", 1),),  # 모르는 세트 이름
])
def test_build_rejects_bad_specs(tmp_path, bad):
    with pytest.raises(ValueError):
        build_dataset(tmp_path, bad, log=lambda *_: None)


# ---------- make_dataset: 재현성 ----------

def tree_bytes(root):
    base = root / DATASET_DIR
    return {p.relative_to(base).as_posix(): p.read_bytes() for p in sorted(base.rglob("*")) if p.is_file()}


def test_same_seeds_give_identical_dataset(tmp_path, dataset_template):
    # 같은 구성을 따로 두 번 만든다: 하나는 세션 시작에 만든 dataset_template(conftest), 하나는 여기서 새로 만든다.
    build_dataset(tmp_path / "b", specs(), log=lambda *_: None)
    assert tree_bytes(tmp_path / "b") == tree_bytes(dataset_template)  # 매니페스트, 요약, 이미지까지 똑같다 (다른 시험이 복사해 쓰는 원본이기도 하다)


def test_different_seeds_give_different_images(tmp_path, dataset_template):
    copy_dataset(dataset_template, tmp_path / "a")  # specs() 로 새로 만든 것과 바이트까지 같다 (위 시험이 확인)
    assert (tmp_path / "a" / DATASET_DIR / MANIFEST).is_file()
    build_dataset(tmp_path / "c", specs(seeds=(11, 12, 13, 14)), log=lambda *_: None)
    a = {r["image_md5"] for r in read_rows(tmp_path / "a")}
    c = {r["image_md5"] for r in read_rows(tmp_path / "c")}
    assert a.isdisjoint(c)


# ---------- make_dataset: 이미 있을 때, 다시 만들기, 삭제 보호 ----------

def test_existing_dataset_is_not_overwritten(small):
    root, _ = small
    before = tree_bytes(root)
    result = build_dataset(root, specs(5, 5, 5, 6), log=lambda *_: None)  # 건수를 바꿔도 건드리지 않는다
    assert result["status"] == "exists"
    assert tree_bytes(root) == before


def test_regenerate_rebuilds_only_the_dataset_dir(tmp_path, dataset_template):
    copy_dataset(dataset_template, tmp_path)  # 처음 데이터셋은 복사 (regenerate 로 다시 만드는 것은 아래에서 실제로 만든다)
    before = tree_bytes(tmp_path)
    (tmp_path / DATASET_DIR / "stray.txt").write_text("남은 파일", encoding="utf-8")
    keep = tmp_path / "data" / "keep.txt"
    keep.write_text("다른 파일", encoding="utf-8")

    messages = []
    result = build_dataset(tmp_path, specs(), regenerate=True, log=messages.append)
    assert result["status"] == "created"
    assert tree_bytes(tmp_path) == before  # stray.txt 는 없어지고 같은 데이터셋이 다시 생김
    assert keep.read_text(encoding="utf-8") == "다른 파일"  # data/dataset 밖은 그대로
    assert any(str(tmp_path / DATASET_DIR) in m for m in messages)  # 지울 경로를 먼저 알린다


def test_incomplete_dataset_needs_regenerate(tmp_path):
    (tmp_path / DATASET_DIR).mkdir(parents=True)
    (tmp_path / DATASET_DIR / "half.png").write_bytes(b"x")
    with pytest.raises(DatasetError, match="--regenerate"):
        build_dataset(tmp_path, specs(), log=lambda *_: None)
    assert (tmp_path / DATASET_DIR / "half.png").exists()  # 옵션 없이는 지우지 않는다
    assert build_dataset(tmp_path, specs(), regenerate=True, log=lambda *_: None)["status"] == "created"
    assert not (tmp_path / DATASET_DIR / "half.png").exists()


def test_remove_refuses_any_path_other_than_data_dataset(tmp_path):
    root = tmp_path / "project"
    outside = tmp_path / "outside"
    others = [root, root / "data", root / DATASET_DIR / "train", root / DATASET_DIR / "..", root / "data" / "other",
              root / "data" / "dataset2", outside]
    for p in others + [root / DATASET_DIR]:
        p.mkdir(parents=True, exist_ok=True)
        (p / "file.txt").write_text("x", encoding="utf-8")

    for p in others:
        with pytest.raises(RefusedPathError):
            remove_dataset(root, p, log=lambda *_: None)
        assert (p / "file.txt").exists()

    assert remove_dataset(root, root / DATASET_DIR, log=lambda *_: None) is True  # 정확히 data/dataset 일 때만 지운다
    assert not (root / DATASET_DIR).exists()
    assert (root / "data").exists() and outside.exists()


# ---------- measure_baseline ----------

def test_parse_seeds():
    assert parse_seeds("42") == (42,)
    assert parse_seeds("42,43,44") == (42, 43, 44)
    for bad in ("", "a", "42,42", "42,,43", "4.2"):
        with pytest.raises(ValueError):
            parse_seeds(bad)


def metrics(acc, rec):
    return ClassMetrics(accuracy=acc, recall=rec, precision=1.0, f1=1.0, confusion_matrix=[[1, 0], [0, 1]], n=2)


def version(normal_acc, drift_acc, drift_rec):
    return {"normal": metrics(normal_acc, 1.0), "drift": metrics(drift_acc, drift_rec)}


def test_gates_boundaries():
    v1 = version(0.80, 0.80, 0.80)
    # 정확히 기준(+1%p, -1%p)인 값은 통과, 조금이라도 모자라면 미통과
    ok = gates(v1, version(0.79, 0.81, 0.79))
    assert [ok[g]["passed"] for g in ("G1", "G2", "G3")] == [True, True, True]
    bad = gates(v1, version(0.789, 0.809, 0.789))
    assert [bad[g]["passed"] for g in ("G1", "G2", "G3")] == [False, False, False]
    assert ok["G1"]["diff"] == pytest.approx(0.01) and ok["G1"]["need"] == 0.01
    assert ok["G2"]["need"] == -0.01 and ok["G3"]["need"] == -0.01


def test_measure_baseline_runs_end_to_end_on_tiny_data(tmp_path, dataset_template):
    copy_dataset(dataset_template, tmp_path)
    config = tmp_path / "predictor.toml"
    config.write_text('threshold = 0.5\nepochs = 1\nbatch_size = 32\nlearning_rate = 0.001\n'
                      'seed = 42\nmodels_dir = "models"\n', encoding="utf-8")
    report_path = tmp_path / "data" / "baseline_report.txt"

    out = run_baseline(tmp_path, seeds=(42, 43), predictor_config=config, report_path=report_path, log=lambda *_: None)

    text = report_path.read_text(encoding="utf-8")
    assert out["report"] == text.rstrip("\n")
    assert "== 시드 42 ==" in text and "== 시드 43 ==" in text and "== 시드 평균" in text
    for ver in ("v1", "v2"):
        for where in ("test_normal", "test_drift", "test_drift dark", "test_drift noise", "test_drift shift"):
            assert f"{ver} {where} 정확도:" in text and f"{ver} {where} 재현율:" in text
            assert f"{ver} {where} 혼동행렬 [[TN, FP], [FN, TP]]:" in text
    for g in ("G1", "G2", "G3"):
        assert any(line.startswith(g) and ("통과" in line or "미통과" in line) for line in text.splitlines())
        assert f"{g} 통과한 시드 수:" in text
    for item in ("v1 학습 시간", "v2 학습 시간", "이미지 1장 추론 시간 평균", "이미지 1장 추론 시간 최대"):
        assert item in text
    assert not (tmp_path / "models").exists()  # 모델은 임시 폴더에만 생기고 지워진다
