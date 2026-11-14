"""추론: 불러온 모델 버전으로 이미지 1장을 판정한다 (docs/data-contract.md 4장)."""
import time
from pathlib import Path

import torch

from common.preprocess import preprocess
from common.schemas import InferenceOutput, Label, validate_image_array
from parts.predictor.config import PROJECT_ROOT, load_config
from parts.predictor.model import SmallCNN
from parts.predictor.storage import ModelLoadError, load_version


class ModelNotLoadedError(RuntimeError):
    """load() 로 모델 버전을 올리기 전에 predict() 를 불렀다."""


class Predictor:
    def __init__(self, config_path=None, *, root=None):
        self.config = load_config(config_path)
        self.root = Path(root) if root is not None else PROJECT_ROOT
        self._models_root = self.root / self.config.models_dir
        self._loaded = None  # (모델, 버전, 임계값). 한 번에 통째로 바꿔서 predict 가 섞인 값을 보지 않게 한다

    @property
    def version(self):
        return self._loaded[1] if self._loaded else None

    @property
    def threshold(self):
        return self._loaded[2] if self._loaded else None

    @property
    def model(self):
        return self._loaded[0] if self._loaded else None

    def load(self, version: str) -> None:
        """models/<version>/ 의 모델을 올린다. 실패하면 ModelLoadError 를 내고 기존 모델을 그대로 둔다."""
        state, metrics = load_version(self._models_root, version)
        model = SmallCNN()
        try:
            model.load_state_dict(state)
        except Exception as e:
            raise ModelLoadError(f"모델 {version} 의 가중치가 모델 구조와 맞지 않습니다: {e}") from e
        model.eval()
        # 임계값은 설정 파일이 아니라 모델 버전에 함께 기록된 값을 쓴다 (계약 1장)
        self._loaded = (model, metrics.version, metrics.threshold)

    def predict(self, image) -> InferenceOutput:
        """이미지 1장(uint8, 128x128) -> {predicted_label, score, model_version, latency_ms}."""
        start = time.perf_counter()
        loaded = self._loaded
        if loaded is None:
            raise ModelNotLoadedError("predict() 전에 load(version) 으로 모델을 올려야 합니다")
        model, version, threshold = loaded
        validate_image_array(image)
        with torch.no_grad():
            score = float(torch.sigmoid(model(preprocess(image))).item())
        label = Label.NG if score >= threshold else Label.OK
        latency_ms = int(round((time.perf_counter() - start) * 1000))
        return InferenceOutput(predicted_label=label, score=score, model_version=version, latency_ms=latency_ms)
