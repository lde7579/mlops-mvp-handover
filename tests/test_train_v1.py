"""scripts/train_v1.py: 고정 데이터셋의 train 으로 첫 모델 models/v1 을 만든다 (작은 데이터셋·임시 폴더만 쓴다).

학습은 시간이 드는 일이라, 한 번 학습한 결과(trained)를 읽기만 하는 시험이 함께 쓰고 학습은 전체 2번만 한다.
"""
import contextlib
import io
import json
import shutil

import numpy as np
import pytest

from backend.database import create_db_engine, init_db, make_session_factory, session_scope
from backend.registry import sync_models
from backend.repository import ModelVersionRepository
from common.schemas import ModelMetrics
from parts.predictor import Predictor
from scripts.make_dataset import DATASET_DIR
from scripts.train_v1 import main

CONFIG = 'threshold = 0.5\nepochs = 1\nbatch_size = 32\nlearning_rate = 0.001\nseed = 42\nmodels_dir = "models"\n'
PROBE = np.full((128, 128), 120, dtype=np.uint8)


def prepare(path, dataset):
    """세션 템플릿(conftest 의 dataset_template: train 10, pool 5, test_normal 5, test_drift 6)을 복사해 설정 파일을 더한다."""
    shutil.copytree(dataset / DATASET_DIR, path / DATASET_DIR)
    (path / "predictor.toml").write_text(CONFIG, encoding="utf-8")
    return path


def run(root):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main([], root=root, predictor_config=root / "predictor.toml")
    return code, out.getvalue()


@pytest.fixture(scope="module")
def trained(tmp_path_factory, dataset_template):
    """v1 을 한 번 학습한 폴더 (학습 1번째)."""
    root = prepare(tmp_path_factory.mktemp("train_v1_trained"), dataset_template)
    code, out = run(root)
    return root, code, out


def test_creates_v1_and_tells_the_next_step(trained):
    root, code, out = trained
    assert code == 0
    assert (root / "models" / "v1" / "model.pt").is_file() and (root / "models" / "v1" / "metrics.json").is_file()
    assert "v1 저장 완료" in out and "시드 42" in out and "python -m backend.registry sync --production v1" in out


def test_metrics_record_train_only_snapshot_and_both_test_sets(trained):
    root, _, _ = trained
    metrics = ModelMetrics.from_dict(json.loads((root / "models" / "v1" / "metrics.json").read_text(encoding="utf-8")))
    assert metrics.version == "v1" and metrics.threshold == 0.5
    assert metrics.train_data_snapshot == {"train": 10, "pool": 0, "seed": 42}  # pool 은 학습에 쓰지 않는다
    assert metrics.normal.n == 5 and metrics.drift.n == 6  # 평가는 test_normal(5장), test_drift(6장)


def test_never_overwrites_an_existing_v1(trained, tmp_path, capsys):
    root, _, _ = trained
    shutil.copytree(root, tmp_path / "copy")
    before = (tmp_path / "copy" / "models" / "v1" / "model.pt").read_bytes()
    assert main([], root=tmp_path / "copy", predictor_config=tmp_path / "copy" / "predictor.toml") == 1
    assert "이미 v1" in capsys.readouterr().err
    assert (tmp_path / "copy" / "models" / "v1" / "model.pt").read_bytes() == before


def test_missing_dataset_says_how_to_make_it_and_creates_nothing(tmp_path, capsys):
    assert main([], root=tmp_path) == 1
    assert "make_dataset" in capsys.readouterr().err
    assert not (tmp_path / "models").exists()


def test_options_are_not_accepted(tmp_path, dataset_template):
    root = prepare(tmp_path, dataset_template)
    assert main(["--epochs", "9"], root=root) == 2
    assert not (root / "models").exists()


def test_same_data_and_settings_give_the_same_model(trained, tmp_path, dataset_template):
    root, _, _ = trained
    other = prepare(tmp_path, dataset_template)
    assert run(other)[0] == 0  # 학습 2번째
    scores = []
    for r in (root, other):
        p = Predictor(config_path=r / "predictor.toml", root=r)
        p.load("v1")
        scores.append(p.predict(PROBE).score)
    assert scores[0] == scores[1]  # 시드 42 고정: 같은 데이터, 같은 설정이면 같은 v1


def test_registry_sync_registers_the_v1_it_made_as_production(trained):
    root, _, _ = trained
    engine = create_db_engine(":memory:")
    try:
        init_db(engine)
        factory = make_session_factory(engine)
        with session_scope(factory) as s:
            report = sync_models(s, root / "models", production="v1")
        assert report.registered == [("v1", "production")]
        with session_scope(factory) as s:
            assert ModelVersionRepository(s).get_production().version == "v1"
    finally:
        engine.dispose()
