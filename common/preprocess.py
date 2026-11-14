"""전처리 — 학습과 추론이 반드시 이 함수 하나를 함께 사용한다 (CLAUDE.md 규칙 5)."""
import numpy as np
import torch

IMG_SIZE = 128


def preprocess(images: np.ndarray) -> torch.Tensor:
    """uint8 이미지 (128,128) 또는 (N,128,128) -> float32 텐서 (N,1,128,128), 값 0~1.

    일부러 이미지별 밝기 보정(정규화)은 하지 않는다. 그래야 '어두워짐' 같은
    현장 변동이 모델에 그대로 전달되어 드리프트가 재현된다.
    """
    arr = np.asarray(images)
    if arr.dtype != np.uint8:
        raise ValueError(f"uint8 이미지가 필요합니다 (받은 타입: {arr.dtype})")
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3 or arr.shape[1:] != (IMG_SIZE, IMG_SIZE):
        raise ValueError(f"128x128 흑백 이미지가 필요합니다 (받은 shape: {arr.shape})")
    return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(1)
