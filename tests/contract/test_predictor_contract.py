"""파트 2(추론) 계약 테스트 — docs/data-contract.md 3장, 4장, 9장."""
from datetime import datetime

import pytest

from common.schemas import InferenceOutput, Label, Prediction
from parts.predictor import Predictor, train

SHORT_CONFIG = """threshold = 0.5
epochs = 2
batch_size = 32
learning_rate = 0.001
seed = 42
models_dir = "models"
"""
OUTPUT_FIELDS = {"predicted_label", "score", "model_version", "latency_ms"}


@pytest.fixture(scope="module")
def predictor(tmp_path_factory, tiny_data):
    root = tmp_path_factory.mktemp("predictor_contract")
    config = root / "predictor.toml"
    config.write_text(SHORT_CONFIG, encoding="utf-8")
    train(*tiny_data.train, version="v1", eval_normal=tiny_data.eval_normal,
          eval_drift=tiny_data.eval_drift, config_path=config, root=root)
    p = Predictor(config_path=config, root=root)
    p.load("v1")
    return p


def test_predictor_io_contract(predictor, tiny_data):
    images = list(tiny_data.eval_normal[0][:5]) + list(tiny_data.eval_drift[0][:5])
    for image in images:
        out = predictor.predict(image)

        # 출력은 {predicted_label, score, model_version, latency_ms} 4개
        assert isinstance(out, InferenceOutput)
        assert set(out.to_dict()) == OUTPUT_FIELDS
        assert out.model_version == "v1"

        # score 는 0~1, 판정 시간은 5초(5000ms) 이내
        assert 0.0 <= out.score <= 1.0
        assert 0 <= out.latency_ms <= 5000

        # 라벨-점수 일치: score >= 임계값이면 NG (1장). Prediction 스키마가 이를 검사한다
        assert out.predicted_label is (Label.NG if out.score >= predictor.threshold else Label.OK)
        Prediction(
            prediction_id="PRD-000001", inspection_id="INS-000001", model_version=out.model_version,
            predicted_label=out.predicted_label, score=out.score, threshold=predictor.threshold,
            latency_ms=out.latency_ms, predicted_at=datetime.now().astimezone(),
        )
