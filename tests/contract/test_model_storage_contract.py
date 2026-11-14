"""모델 저장 형식 계약 테스트 — docs/data-contract.md 4장(metrics.json), 9장."""
import json

import torch

from common.schemas import ModelMetrics
from parts.predictor import train
from parts.predictor.model import SmallCNN

SHORT_CONFIG = """threshold = 0.5
epochs = 2
batch_size = 32
learning_rate = 0.001
seed = 42
models_dir = "models"
"""


def test_model_storage_contract(tmp_path, tiny_data):
    config = tmp_path / "predictor.toml"
    config.write_text(SHORT_CONFIG, encoding="utf-8")
    train(*tiny_data.train, version="v1", eval_normal=tiny_data.eval_normal,
          eval_drift=tiny_data.eval_drift, config_path=config, root=tmp_path)

    folder = tmp_path / "models" / "v1"
    assert (folder / "model.pt").is_file()
    assert (folder / "metrics.json").is_file()

    # metrics.json 이 4장의 필드 형식과 맞는다 (ModelMetrics 가 필드와 형식을 모두 검사)
    metrics = ModelMetrics.from_dict(json.loads((folder / "metrics.json").read_text(encoding="utf-8")))
    assert metrics.version == "v1"
    assert metrics.threshold == 0.5
    assert metrics.normal.n == len(tiny_data.eval_normal[1])
    assert metrics.drift.n == len(tiny_data.eval_drift[1])
    for part in (metrics.normal, metrics.drift):
        assert sum(sum(row) for row in part.confusion_matrix) == part.n
    snap = metrics.train_data_snapshot
    assert snap["train"] + snap["pool"] == len(tiny_data.train[1])

    # model.pt 는 가중치만 담겨 있어 weights_only 로 안전하게 읽히고 모델 구조에 들어간다
    SmallCNN().load_state_dict(torch.load(folder / "model.pt", weights_only=True, map_location="cpu"))

    # 저장 도중 쓰는 임시 폴더가 남아 있지 않다
    assert sorted(p.name for p in (tmp_path / "models").iterdir()) == ["v1"]
