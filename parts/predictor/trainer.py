"""학습(train)과 평가(evaluate). 전처리는 common/preprocess.py 한 곳만 쓴다 (학습·추론 공유)."""
import hashlib
import random
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn

from common.preprocess import preprocess
from common.schemas import IMAGE_SIZE, ClassMetrics, Label, ModelMetrics, SchemaError
from parts.predictor.config import PROJECT_ROOT, load_config
from parts.predictor.model import SmallCNN
from parts.predictor.storage import check_version, save_version, version_dir

POS_WEIGHT = 2.0  # 불량(NG)을 놓치는 쪽이 더 치명적이라 NG 쪽에 가중치 (스파이크와 같은 값)
_EVAL_BATCH = 256


@contextmanager
def _deterministic():
    """학습 동안만 결정적 연산을 켜고, 끝나면 원래 설정으로 되돌린다."""
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _check_images(name: str, images) -> np.ndarray:
    arr = np.asarray(images)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE) or len(arr) < 1:
        raise SchemaError(
            f"{name}: uint8 이미지 배열 (N, {IMAGE_SIZE}, {IMAGE_SIZE}), N >= 1 이어야 합니다 "
            f"(받은 값: dtype={arr.dtype}, shape={arr.shape})")
    return arr


def _check_labels(name: str, labels, n: int) -> np.ndarray:
    """'OK'/'NG' 목록 -> 0/1 배열 (NG = 1)."""
    values = []
    for v in labels:
        try:
            values.append(1 if Label(v) is Label.NG else 0)
        except ValueError:
            raise SchemaError(f"{name}: 라벨은 OK 또는 NG 여야 합니다 (받은 값: {v!r})") from None
    if len(values) != n:
        raise SchemaError(f"{name}: 라벨 개수({len(values)})가 이미지 개수({n})와 다릅니다")
    return np.array(values, dtype=np.int64)


def _digests(images: np.ndarray) -> set:
    return {hashlib.md5(im.tobytes()).hexdigest() for im in images}


def _check_snapshot(snapshot, n: int) -> dict:
    if snapshot is None:
        return {"train": n, "pool": 0}
    if (not isinstance(snapshot, dict) or set(snapshot) != {"train", "pool"}
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in snapshot.values())):
        raise ValueError(f"snapshot: {{'train': 건수, 'pool': 건수}} 형태여야 합니다 (받은 값: {snapshot!r})")
    if snapshot["train"] + snapshot["pool"] != n:
        raise ValueError(f"snapshot: train + pool({snapshot['train'] + snapshot['pool']})이 학습 이미지 개수({n})와 다릅니다")
    return {"train": snapshot["train"], "pool": snapshot["pool"]}


@torch.no_grad()
def _scores(model: nn.Module, images: np.ndarray) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(preprocess(images[i:i + _EVAL_BATCH]))).numpy()
           for i in range(0, len(images), _EVAL_BATCH)]
    return np.concatenate(out)


def evaluate(model, images, labels, threshold: float) -> ClassMetrics:
    """한 평가셋의 지표. model 은 SmallCNN(nn.Module) 또는 Predictor (.model 을 가진 객체)."""
    net = getattr(model, "model", model)
    arr = _check_images("images", images)
    y = _check_labels("labels", labels, len(arr))
    pred = (_scores(net, arr) >= threshold).astype(int)
    return ClassMetrics(
        accuracy=float(accuracy_score(y, pred)),
        recall=float(recall_score(y, pred, zero_division=0)),
        precision=float(precision_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist(),  # [[TN, FP], [FN, TP]]
        n=int(len(y)),
    )


def train(images, labels, *, version, eval_normal, eval_drift, snapshot=None, seed=None,
          config_path=None, root=None) -> ModelMetrics:
    """모델을 학습해 models/<version>/ 에 model.pt, metrics.json 으로 저장하고 ModelMetrics 를 돌려준다.

    eval_normal, eval_drift: (이미지 배열, 라벨 목록). 호출하는 쪽이 고정 테스트셋을 넘긴다.
    snapshot: {"train": 건수, "pool": 건수}. 생략하면 {"train": N, "pool": 0}. 기록할 때 seed 를 덧붙인다.
    """
    cfg = load_config(config_path)
    models_root = (PROJECT_ROOT if root is None else Path(root)) / cfg.models_dir
    check_version(version)
    if seed is None:
        seed = cfg.seed
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed: 정수여야 합니다 (받은 값: {seed!r})")
    if version_dir(models_root, version).exists():
        raise FileExistsError(f"이미 있는 버전입니다: {version}")

    x_img = _check_images("images", images)
    y = _check_labels("labels", labels, len(x_img))
    normal_img = _check_images("eval_normal 이미지", eval_normal[0])
    normal_y = _check_labels("eval_normal 라벨", eval_normal[1], len(normal_img))
    drift_img = _check_images("eval_drift 이미지", eval_drift[0])
    drift_y = _check_labels("eval_drift 라벨", eval_drift[1], len(drift_img))
    snap = _check_snapshot(snapshot, len(x_img))

    # 평가용 데이터를 학습에 쓰면 안 된다 (CLAUDE.md 규칙 6)
    if _digests(x_img) & (_digests(normal_img) | _digests(drift_img)):
        raise ValueError("평가 이미지가 학습 이미지와 겹칩니다. 평가셋은 학습에 쓸 수 없습니다")

    with _deterministic():
        _set_seed(seed)
        model = SmallCNN()
        x = preprocess(x_img)
        target = torch.from_numpy(y.astype(np.float32))
        opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT))
        gen = torch.Generator().manual_seed(seed)
        model.train()
        for _ in range(cfg.epochs):
            perm = torch.randperm(len(x), generator=gen)
            for i in range(0, len(x), cfg.batch_size):
                idx = perm[i:i + cfg.batch_size]
                opt.zero_grad()
                loss_fn(model(x[idx]), target[idx]).backward()
                opt.step()
    model.eval()

    metrics = ModelMetrics(
        version=version,
        threshold=cfg.threshold,
        normal=evaluate(model, normal_img, [Label.NG if v else Label.OK for v in normal_y], cfg.threshold),
        drift=evaluate(model, drift_img, [Label.NG if v else Label.OK for v in drift_y], cfg.threshold),
        train_data_snapshot={**snap, "seed": seed},
        created_at=datetime.now().astimezone(),
    )
    save_version(models_root, version, model.state_dict(), metrics)
    return metrics
