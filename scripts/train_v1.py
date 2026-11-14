"""v1 학습: 고정 데이터셋(data/dataset)의 train 으로 첫 모델 models/v1 을 만든다.

실행:  python -m scripts.train_v1
그다음:  python -m backend.registry sync --production v1     (DB 에 운영 모델로 등록)

- 학습 설정(에폭, 학습률, 임계값, 시드 42)은 parts/predictor/predictor.toml 그대로 쓴다. 같은 데이터, 같은 설정이면 같은 v1 이 나온다.
- 평가는 data/dataset 의 test_normal, test_drift 로 하고 학습에는 쓰지 않는다 (train 만 학습).
- 이미 models/v1 이 있으면 덮어쓰지 않고 멈춘다 (지우려면 사용자가 직접 지운다).
- 데이터셋이 없으면 먼저 python -m scripts.make_dataset 을 실행하라고 알린다.
"""
import sys
import time
from pathlib import Path

from parts.predictor import train
from scripts.make_dataset import PROJECT_ROOT, DatasetError, load_dataset

VERSION = "v1"


def train_v1(root=PROJECT_ROOT, predictor_config=None, log=print):
    root = Path(root)
    data, _ = load_dataset(root)
    train_set = data["train"]
    log(f"학습 데이터: train {len(train_set.labels)}장 (평가: test_normal {len(data['test_normal'].labels)}장, "
        f"test_drift {len(data['test_drift'].labels)}장)")
    start = time.perf_counter()
    metrics = train(
        train_set.images, train_set.labels, version=VERSION,
        eval_normal=(data["test_normal"].images, data["test_normal"].labels),
        eval_drift=(data["test_drift"].images, data["test_drift"].labels),
        snapshot={"train": len(train_set.labels), "pool": 0},
        config_path=predictor_config, root=root)
    log(f"학습 시간: {time.perf_counter() - start:.1f}초")
    return metrics


def main(argv=None, root=PROJECT_ROOT, predictor_config=None) -> int:
    if argv:
        print("옵션이 없습니다. 사용법: python -m scripts.train_v1", file=sys.stderr)
        return 2
    try:
        metrics = train_v1(root, predictor_config)
    except DatasetError as e:
        print(f"실패: {e}", file=sys.stderr)
        return 1
    except FileExistsError:
        print(f"실패: 이미 {VERSION} 모델이 있습니다. 덮어쓰지 않았습니다 (다시 만들려면 models/{VERSION} 폴더를 직접 지우세요)", file=sys.stderr)
        return 1
    print(f"{VERSION} 저장 완료 (임계값 {metrics.threshold}, 시드 {metrics.train_data_snapshot['seed']})")
    for name, m in (("test_normal", metrics.normal), ("test_drift", metrics.drift)):
        print(f"{name} 정확도 {m.accuracy:.3f}, 재현율 {m.recall:.3f}")
    print("다음: python -m backend.registry sync --production v1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
