"""backend/registry.py: models/<version>/ 폴더(파일)를 model_version 테이블(DB)에 등록하는 동기화."""
import json

import pytest

from backend.database import session_scope
from backend.registry import RegistryError, main, models_root, sync_models
from backend.repository import ModelVersionRepository, Stage
from tests.backend.helpers import clone_version, copy_models, make_settings


@pytest.fixture
def root(tmp_path, trained_models):
    copy_models(trained_models.root, tmp_path)  # models/v1
    return tmp_path


def sync(factory, root, **kw):
    with session_scope(factory) as s:
        return sync_models(s, root / "models", **kw)


def record(factory, version):
    with session_scope(factory) as s:
        return ModelVersionRepository(s).get(version)


# ---------- 등록 ----------

def test_missing_or_empty_models_folder_registers_nothing(factory, tmp_path):
    for target in (tmp_path / "nope", tmp_path):
        with session_scope(factory) as s:
            report = sync_models(s, target)
        assert report.registered == [] and report.unchanged == [] and report.skipped == []


def test_new_version_is_registered_as_candidate_by_default(factory, root, trained_models):
    report = sync(factory, root)
    assert report.registered == [("v1", "candidate")]
    got = record(factory, "v1")
    assert got.stage is Stage.CANDIDATE and got.approved_by is None and got.approved_at is None
    assert got.metrics.to_dict() == trained_models.metrics.to_dict()  # 파일의 metrics.json 과 같다
    assert got.train_data_snapshot == trained_models.metrics.train_data_snapshot


def test_production_option_registers_that_version_as_production(factory, root):
    report = sync(factory, root, production="v1")
    assert report.registered == [("v1", "production")]
    assert record(factory, "v1").stage is Stage.PRODUCTION


def test_sync_twice_changes_nothing(factory, root):
    sync(factory, root, production="v1")
    report = sync(factory, root, production="v1")  # 이미 production 인 같은 버전을 다시 지정해도 괜찮다
    assert report.registered == [] and report.unchanged == ["v1"]
    assert record(factory, "v1").stage is Stage.PRODUCTION
    again = sync(factory, root)
    assert again.registered == [] and again.unchanged == ["v1"]


def test_versions_are_registered_in_numeric_order(factory, root):
    clone_version(root, "v1", "v10")
    clone_version(root, "v1", "v2")
    assert [v for v, _ in sync(factory, root).registered] == ["v1", "v2", "v10"]  # 글자 순서(v1, v10, v2)가 아니다


def test_new_file_is_added_later_and_old_one_is_left_alone(factory, root):
    sync(factory, root, production="v1")
    clone_version(root, "v1", "v2")
    report = sync(factory, root)
    assert report.registered == [("v2", "candidate")] and report.unchanged == ["v1"]
    assert record(factory, "v1").stage is Stage.PRODUCTION  # 기존 production 은 그대로


# ---------- --production 거부 (아무것도 바꾸지 않는다) ----------

def test_production_is_refused_when_another_production_exists(factory, root):
    sync(factory, root, production="v1")
    clone_version(root, "v1", "v2")
    with pytest.raises(RegistryError, match="v1"):
        sync(factory, root, production="v2")
    assert record(factory, "v2") is None  # v2 는 candidate 로도 등록되지 않았다
    assert record(factory, "v1").stage is Stage.PRODUCTION


def test_production_is_refused_for_a_version_registered_as_something_else(factory, root):
    sync(factory, root)  # v1 = candidate
    with pytest.raises(RegistryError, match="candidate"):
        sync(factory, root, production="v1")
    assert record(factory, "v1").stage is Stage.CANDIDATE


@pytest.mark.parametrize("version", ["v9", "v1x", "../v1", ""])
def test_production_is_refused_when_there_is_no_such_model_file(factory, root, version):
    with pytest.raises(RegistryError):
        sync(factory, root, production=version)
    assert record(factory, "v1") is None


# ---------- 파일과 DB 가 어긋날 때 ----------

def test_db_row_without_files_is_reported_and_kept(factory, root):
    sync(factory, root, production="v1")
    for name in ("model.pt", "metrics.json"):
        (root / "models" / "v1" / name).unlink()
    report = sync(factory, root)
    assert report.missing_files == ["v1"] and report.registered == []
    assert record(factory, "v1").stage is Stage.PRODUCTION  # 지우지 않는다
    assert any("파일이 없음" in line and "v1" in line for line in report.lines())


def test_only_model_pt_missing_counts_as_missing_files(factory, root):
    sync(factory, root)
    (root / "models" / "v1" / "model.pt").unlink()
    assert sync(factory, root).missing_files == ["v1"]


def test_files_that_differ_from_the_db_are_reported_not_overwritten(factory, root):
    sync(factory, root)
    clone_version(root, "v1", "v2")
    sync(factory, root)
    path = root / "models" / "v2" / "metrics.json"
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["threshold"] = 0.9
    path.write_text(json.dumps(changed), encoding="utf-8")
    before = record(factory, "v2").metrics.to_dict()
    report = sync(factory, root)
    assert report.mismatched == ["v2"] and report.unchanged == ["v1", "v2"]
    assert record(factory, "v2").metrics.to_dict() == before and before["threshold"] == 0.5


# ---------- 건너뛰는 폴더 ----------

@pytest.mark.parametrize("name", ["v1x", "V3", "backup", ".tmp-v2-abc123", "v", "v-1"])
def test_folders_that_are_not_version_folders_are_skipped(factory, root, name):
    (root / "models" / name).mkdir()
    report = sync(factory, root)
    assert [n for n, _ in report.skipped] == [name] and report.registered == [("v1", "candidate")]


def test_plain_files_in_models_folder_are_ignored(factory, root):
    (root / "models" / "notes.txt").write_text("x", encoding="utf-8")
    report = sync(factory, root)
    assert report.skipped == [] and report.registered == [("v1", "candidate")]


def test_folder_missing_a_file_is_skipped_and_others_still_register(factory, root):
    clone_version(root, "v1", "v2")
    (root / "models" / "v2" / "model.pt").unlink()
    report = sync(factory, root)
    assert report.registered == [("v1", "candidate")] and [n for n, _ in report.skipped] == ["v2"]
    assert record(factory, "v2") is None


@pytest.mark.parametrize("content", ["", "not json", "[]", '{"version": "v2"}'])
def test_broken_metrics_json_is_skipped_with_a_reason(factory, root, content):
    clone_version(root, "v1", "v2")
    (root / "models" / "v2" / "metrics.json").write_text(content, encoding="utf-8")
    report = sync(factory, root)
    assert report.registered == [("v1", "candidate")]
    assert [n for n, _ in report.skipped] == ["v2"] and "metrics.json" in report.skipped[0][1]


def test_metrics_version_must_match_the_folder_name(factory, root):
    clone_version(root, "v1", "v2")
    path = root / "models" / "v2" / "metrics.json"
    path.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")), "version": "v1"}), encoding="utf-8")
    report = sync(factory, root)
    assert [n for n, _ in report.skipped] == ["v2"] and "v2" in report.skipped[0][1]


# ---------- 모델 폴더 위치: Predictor 가 읽는 곳과 같아야 한다 (parts/predictor 설정 의존 보호) ----------

def test_models_root_follows_the_predictor_config(tmp_path):
    from parts.predictor.config import PredictorConfig, load_config  # 이 함수와 models_dir 항목에 의존한다

    config = load_config()
    assert isinstance(config, PredictorConfig) and isinstance(config.models_dir, str), \
        "parts.predictor.config.load_config 가 바뀌었습니다 -> backend/registry.py 의 models_root() 를 고쳐야 합니다"
    assert models_root(tmp_path) == tmp_path / config.models_dir


def test_predictor_loads_from_the_folder_the_registry_scans(tmp_path, trained_models):
    from parts.predictor import Predictor

    target = models_root(tmp_path)
    target.mkdir(parents=True)
    import shutil
    shutil.copytree(trained_models.root / "models" / "v1", target / "v1")
    predictor = Predictor(root=tmp_path)  # 설정 파일은 기본값(실제 predictor.toml)
    predictor.load("v1")
    assert predictor.version == "v1", "Predictor 가 읽는 모델 폴더와 레지스트리가 보는 폴더가 달라졌습니다"


# ---------- 명령줄 ----------

def test_cli_registers_and_prints_a_report(tmp_path, root, trained_models, capsys):
    settings = make_settings(root=str(root), db_path=str(tmp_path / "cli.db"))
    lines = []
    assert main(["sync", "--production", "v1"], settings=settings, out=lines.append) == 0
    assert lines[0] == "새로 등록: v1(production)"
    lines2 = []
    assert main(["sync"], settings=settings, out=lines2.append) == 0
    assert lines2[0] == "새로 등록: 없음" and "v1" in lines2[1]


def test_cli_refusal_returns_1_and_writes_nothing(tmp_path, root, capsys):
    settings = make_settings(root=str(root), db_path=str(tmp_path / "cli.db"))
    assert main(["sync", "--production", "v9"], settings=settings) == 1
    assert "v9" in capsys.readouterr().err
    lines = []
    main(["sync"], settings=settings, out=lines.append)
    assert lines[0] == "새로 등록: v1(candidate)"  # 앞의 실패가 아무것도 남기지 않았다


def test_cli_requires_a_command(capsys):
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == 2
