"""테스트 공용 도구: 학습·평가용 작은 데이터 생성만 한다 (parts.collector.Collector 로 tmp 폴더에 만든다)."""
from dataclasses import dataclass

import numpy as np
import pytest

from parts.collector import Collector


@dataclass
class TinyData:
    train: tuple         # (이미지 (60,128,128), 라벨 목록)
    eval_normal: tuple   # (이미지 (20,128,128), 라벨 목록) 정상 환경
    eval_drift: tuple    # (이미지 (20,128,128), 라벨 목록) 드리프트 환경


def _make(root, seed, n, scenario):
    collector = Collector(root=root, seed=seed)  # 세트마다 시드를 달리해 이미지가 겹치지 않게 한다
    images, labels = [], []
    for i in range(n):
        label = "NG" if i % 3 == 0 else "OK"
        sample = collector.next_sample(scenario=scenario, label=label)
        images.append(sample.image)
        labels.append(label)
    return np.stack(images), labels


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory):
    """학습·평가용 작은 이미지 배열 (60 + 20 + 20장).

    이 fixture 를 쓰는 시험은 이미지 배열만 읽고 PNG 파일은 읽지 않는다. 그래서 만드는 동안만 수집기의 PNG 저장·재검사를 생략한다
    (이미지 1장 시간의 약 58%; 저장은 난수를 쓰지 않아 이미지 내용은 한 바이트도 달라지지 않는다).
    수집기의 내부 메서드 _save_png 에 의존하므로, 이름이 바뀌면 여기서 바로 실패한다. 저장·재검사는 tests/test_collector.py 와
    tests/contract/test_collector_contract.py 가 확인한다.
    """
    root = tmp_path_factory.mktemp("tiny_data")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Collector, "_save_png", lambda self, inspection_id, path, image: None)
        return TinyData(
            train=_make(root / "train", seed=101, n=60, scenario="normal"),
            eval_normal=_make(root / "eval_normal", seed=102, n=20, scenario="normal"),
            eval_drift=_make(root / "eval_drift", seed=103, n=20, scenario="drift"),
        )


@pytest.fixture(scope="session")
def dataset_template(tmp_path_factory):
    """작은 고정 데이터셋(26장: train 10, pool 5, test_normal 5, test_drift 6, 시드 1~4)을 세션당 한 번만 만든다.

    데이터셋이 필요한 시험(scripts, sweep, train_v1)은 이것을 자기 임시 폴더로 복사해서 쓴다 (원본은 읽기 전용).
    같은 구성을 새로 만든 것과 바이트까지 같다는 것은 tests/test_scripts.py::test_same_seeds_give_identical_dataset 이 매번 확인한다.
    """
    from scripts.make_dataset import SplitSpec, build_dataset

    root = tmp_path_factory.mktemp("dataset_template")
    specs = (SplitSpec("train", 10, "normal", 1), SplitSpec("pool", 5, "drift", 2),
             SplitSpec("test_normal", 5, "normal", 3), SplitSpec("test_drift", 6, "drift", 4))
    build_dataset(root, specs, log=lambda *_: None)
    return root
