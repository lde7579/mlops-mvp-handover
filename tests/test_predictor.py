"""parts/predictor 보조 테스트 (계약 테스트는 tests/contract/ 에 있다)."""
import json
import shutil

import numpy as np
import pytest
import torch
from torch import nn

from common.schemas import ClassMetrics, InferenceOutput, Label, SchemaError
from parts.predictor import ModelLoadError, ModelNotLoadedError, Predictor, evaluate, train
from parts.predictor import storage as storage_module
from parts.predictor.config import ConfigError, load_config
from parts.predictor.storage import load_version

BASE = {"threshold": 0.5, "epochs": 2, "batch_size": 32, "learning_rate": 0.001, "seed": 42, "models_dir": "models"}


def toml_text(**over):
    values = {**BASE, **over}
    lines = []
    for key, v in values.items():
        if v is None:  # 항목을 지운다
            continue
        text = ("true" if v else "false") if isinstance(v, bool) else json.dumps(v)
        lines.append(f"{key} = {text}")
    return "\n".join(lines) + "\n"


def write_config(path, **over):
    path.write_text(toml_text(**over), encoding="utf-8")
    return path


def run_train(tiny, root, config, version="v1", **kw):
    return train(*tiny.train, version=version, eval_normal=tiny.eval_normal, eval_drift=tiny.eval_drift,
                 config_path=config, root=root, **kw)


@pytest.fixture(scope="module")
def env(tmp_path_factory, tiny_data):
    """v1 하나를 학습해 두고, 읽기만 하는 테스트가 함께 쓴다."""
    root = tmp_path_factory.mktemp("predictor_env")
    config = write_config(root / "predictor.toml")
    metrics = run_train(tiny_data, root, config)
    predictor = Predictor(config_path=config, root=root)
    predictor.load("v1")
    return {"root": root, "config": config, "metrics": metrics, "predictor": predictor}


# ---------- 설정 ----------

def test_default_config_values():
    cfg = load_config()
    assert (cfg.threshold, cfg.epochs, cfg.batch_size) == (0.5, 5, 64)
    assert (cfg.learning_rate, cfg.seed, cfg.models_dir) == (0.001, 42, "models")


@pytest.mark.parametrize(
    "over",
    [
        {"threshold": 1.5}, {"threshold": -0.1}, {"threshold": "0.5"}, {"threshold": True},
        {"epochs": 0}, {"epochs": 2.5}, {"epochs": True},
        {"batch_size": 0}, {"batch_size": "64"},
        {"learning_rate": 0}, {"learning_rate": -0.1}, {"learning_rate": "0.001"},
        {"seed": 4.2}, {"seed": "42"},
        {"models_dir": "/models"}, {"models_dir": "C:\\models"}, {"models_dir": ""}, {"models_dir": 3},
        {"seed": None},  # 빠진 항목
        {"unknown_key": 1},  # 모르는 항목
    ],
)
def test_config_rejects_bad_values(tmp_path, over):
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path / "bad.toml", **over))


# ---------- 추론 ----------

def test_predict_returns_valid_output(env, tiny_data):
    p = env["predictor"]
    assert p.version == "v1" and p.threshold == 0.5
    for image in tiny_data.eval_normal[0][:5]:
        out = p.predict(image)
        assert isinstance(out, InferenceOutput) and isinstance(out.latency_ms, int)
        assert out.model_version == "v1"
        assert out.predicted_label is (Label.NG if out.score >= p.threshold else Label.OK)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((64, 64), dtype=np.uint8),
        np.zeros((128, 128, 3), dtype=np.uint8),
        np.zeros((1, 128, 128), dtype=np.uint8),
        np.zeros((128, 128), dtype=np.float32),
        [[0] * 128] * 128,
        None,
    ],
)
def test_predict_rejects_bad_image(env, image):
    with pytest.raises(SchemaError):
        env["predictor"].predict(image)


def test_predict_requires_loaded_model(env, tiny_data):
    fresh = Predictor(config_path=env["config"], root=env["root"])
    assert fresh.version is None and fresh.threshold is None
    with pytest.raises(ModelNotLoadedError):
        fresh.predict(tiny_data.eval_normal[0][0])


def test_saved_model_gives_same_output_after_reload(env, tiny_data):
    reloaded = Predictor(config_path=env["config"], root=env["root"])
    reloaded.load("v1")
    for image in tiny_data.eval_normal[0][:5]:
        a, b = env["predictor"].predict(image), reloaded.predict(image)
        assert (a.score, a.predicted_label) == (b.score, b.predicted_label)


def test_failed_load_keeps_current_model(env, tiny_data, tmp_path):
    root = tmp_path / "root"
    shutil.copytree(env["root"] / "models", root / "models")
    p = Predictor(config_path=env["config"], root=root)
    p.load("v1")
    image = tiny_data.eval_normal[0][0]
    before = p.predict(image)

    metrics = json.loads((root / "models" / "v1" / "metrics.json").read_text(encoding="utf-8"))
    (root / "models" / "v2").mkdir()  # 깨진 가중치
    (root / "models" / "v2" / "model.pt").write_bytes(b"not a model")
    (root / "models" / "v2" / "metrics.json").write_text(json.dumps({**metrics, "version": "v2"}), encoding="utf-8")
    shutil.copytree(root / "models" / "v1", root / "models" / "v3")  # metrics 의 version 이 폴더 이름과 다름

    for bad in ("v9", "v2", "v3"):  # 없는 버전, 깨진 파일, 이름 불일치
        with pytest.raises(ModelLoadError):
            p.load(bad)
        assert p.version == "v1"
        again = p.predict(image)
        assert (again.score, again.model_version) == (before.score, "v1")
    with pytest.raises(ValueError):  # 폴더 밖을 가리키는 이름
        p.load("../v1")


# ---------- 학습 ----------

def test_same_seed_same_data_gives_same_model(env, tmp_path, tiny_data):
    before = torch.are_deterministic_algorithms_enabled()
    # "a" 는 모듈 fixture env 가 이미 학습해 둔 v1 이다 (같은 설정, 시드 42, 같은 데이터). b 는 같은 조건으로 새로 학습하고,
    # c 는 시드만 다르게 학습한다.
    roots = {"a": load_version(env["root"] / "models", "v1")}
    for name, seed in (("b", 42), ("c", 7)):
        root = tmp_path / name
        root.mkdir()
        config = write_config(root / "predictor.toml")
        run_train(tiny_data, root, config, seed=seed)
        roots[name] = load_version(root / "models", "v1")
    assert torch.are_deterministic_algorithms_enabled() == before  # 학습이 끝나면 원래 설정으로 복원

    (state_a, metrics_a), (state_b, metrics_b), (state_c, _) = roots["a"], roots["b"], roots["c"]
    assert state_a.keys() == state_b.keys()
    assert all(torch.equal(state_a[k], state_b[k]) for k in state_a)
    assert (metrics_a.normal, metrics_a.drift) == (metrics_b.normal, metrics_b.drift)
    assert any(not torch.equal(state_a[k], state_c[k]) for k in state_a)  # 시드가 다르면 달라짐


def test_threshold_and_snapshot_are_recorded(tmp_path, tiny_data):
    config = write_config(tmp_path / "predictor.toml", threshold=0.7, epochs=1)  # 임계값·snapshot 기록은 학습 횟수와 무관
    metrics = run_train(tiny_data, tmp_path, config, snapshot={"train": 40, "pool": 20})
    assert metrics.threshold == 0.7
    assert metrics.train_data_snapshot == {"train": 40, "pool": 20, "seed": 42}

    p = Predictor(config_path=config, root=tmp_path)
    p.load("v1")
    assert p.threshold == 0.7  # 설정 파일이 아니라 모델에 기록된 임계값
    for image in tiny_data.eval_normal[0]:
        out = p.predict(image)
        assert out.predicted_label is (Label.NG if out.score >= 0.7 else Label.OK)


def test_default_snapshot_is_all_train(env):
    assert env["metrics"].train_data_snapshot == {"train": 60, "pool": 0, "seed": 42}


def test_existing_version_is_not_overwritten(env, tiny_data):
    before = (env["root"] / "models" / "v1" / "model.pt").read_bytes()
    with pytest.raises(FileExistsError):
        run_train(tiny_data, env["root"], env["config"])
    assert (env["root"] / "models" / "v1" / "model.pt").read_bytes() == before


@pytest.mark.parametrize("version", ["../x", "V1", "1", "v", "v1.0", "", " v1", "v1/..", None, 1])
def test_bad_version_name_is_rejected(tmp_path, tiny_data, version):
    config = write_config(tmp_path / "predictor.toml")
    with pytest.raises(ValueError):
        run_train(tiny_data, tmp_path, config, version=version)
    assert not (tmp_path / "models").exists()


def test_eval_set_overlapping_train_is_rejected(tmp_path, tiny_data):
    config = write_config(tmp_path / "predictor.toml")
    leaked = (tiny_data.train[0][:10], tiny_data.train[1][:10])
    with pytest.raises(ValueError, match="겹칩니다"):
        train(*tiny_data.train, version="v1", eval_normal=leaked, eval_drift=tiny_data.eval_drift,
              config_path=config, root=tmp_path)
    assert not (tmp_path / "models").exists()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: {"labels": ["BAD"] + d.train[1][1:]},
        lambda d: {"labels": d.train[1][:-1]},  # 개수 불일치
        lambda d: {"images": d.train[0].astype(np.float32)},
        lambda d: {"images": d.train[0][:, :64, :64]},
    ],
)
def test_bad_training_data_is_rejected(tmp_path, tiny_data, mutate):
    config = write_config(tmp_path / "predictor.toml")
    args = {"images": tiny_data.train[0], "labels": tiny_data.train[1], **mutate(tiny_data)}
    with pytest.raises(SchemaError):
        train(args["images"], args["labels"], version="v1", eval_normal=tiny_data.eval_normal,
              eval_drift=tiny_data.eval_drift, config_path=config, root=tmp_path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"snapshot": {"train": 10, "pool": 10}},  # 합이 학습 개수(60)와 다름
        {"snapshot": {"train": 60}},
        {"snapshot": {"train": -1, "pool": 61}},
        {"snapshot": {"train": 60, "pool": 0, "extra": 1}},
        {"seed": "42"},
        {"seed": True},
    ],
)
def test_bad_snapshot_or_seed_is_rejected(tmp_path, tiny_data, kwargs):
    config = write_config(tmp_path / "predictor.toml")
    with pytest.raises(ValueError):
        run_train(tiny_data, tmp_path, config, **kwargs)


def test_failed_save_leaves_no_folder(tmp_path, tiny_data, monkeypatch):
    config = write_config(tmp_path / "predictor.toml", epochs=1)  # 저장 실패 처리는 학습 횟수와 무관

    def disk_full(obj, path):
        raise OSError("disk full")

    monkeypatch.setattr(storage_module.torch, "save", disk_full)
    with pytest.raises(OSError):
        run_train(tiny_data, tmp_path, config)
    assert list((tmp_path / "models").iterdir()) == []  # 임시 폴더도, 반쪽짜리 v1 도 없음


# ---------- 평가 ----------

class TableModel(nn.Module):
    """이미지 전체가 값 k 로 채워져 있으면 table[k] 확률을 내는 가짜 모델 (지표 계산 확인용)."""

    def __init__(self, table):
        super().__init__()
        self.logits = torch.logit(torch.tensor(table, dtype=torch.float32))

    def forward(self, x):
        return self.logits[(x.mean(dim=(1, 2, 3)) * 255).round().long()]


def test_evaluate_matches_hand_calculation():
    images = np.stack([np.full((128, 128), k, dtype=np.uint8) for k in range(6)])
    labels = ["OK", "NG", "NG", "OK", "OK", "NG"]
    model = TableModel([0.1, 0.9, 0.4, 0.6, 0.2, 0.8])  # 0.5 기준 판정: OK NG OK NG OK NG
    m = evaluate(model, images, labels, 0.5)
    assert m.confusion_matrix == [[2, 1], [1, 2]]  # [[TN, FP], [FN, TP]]
    assert m.n == 6
    assert m.accuracy == pytest.approx(4 / 6)
    assert m.recall == pytest.approx(2 / 3)
    assert m.precision == pytest.approx(2 / 3)
    assert m.f1 == pytest.approx(2 / 3)
    assert evaluate(model, images, labels, 0.95).confusion_matrix == [[3, 0], [3, 0]]  # 임계값을 올리면 전부 OK


def test_evaluate_with_predictor_matches_saved_metrics(env, tiny_data):
    p = env["predictor"]
    again = evaluate(p, *tiny_data.eval_normal, p.threshold)
    assert isinstance(again, ClassMetrics)
    assert again == env["metrics"].normal
