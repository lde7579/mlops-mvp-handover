"""백엔드 테스트 공용 fixture: 테스트마다 새 메모리 DB (테이블 7개라 만드는 데 수 ms)."""
import pytest

from backend.database import create_db_engine, init_db, make_session_factory


@pytest.fixture
def engine():
    e = create_db_engine(":memory:")
    init_db(e)
    yield e
    e.dispose()


@pytest.fixture
def factory(engine):
    return make_session_factory(engine)


@pytest.fixture
def session(factory):
    s = factory()
    yield s
    s.rollback()
    s.close()


@pytest.fixture
def file_engine(tmp_path):
    e = create_db_engine(tmp_path / "test.db")
    init_db(e)
    yield e
    e.dispose()


# ---------- 3-B: 앱·클라이언트 ----------
from fastapi.testclient import TestClient

from backend.main import create_app
from tests.backend.helpers import make_settings


@pytest.fixture
def settings():
    return make_settings()


@pytest.fixture
def app(settings):
    a = create_app(settings)
    yield a
    a.state.engine.dispose()


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


# ---------- 3-D: 학습해 둔 작은 모델 (세션당 1회) ----------
from types import SimpleNamespace


def _noise_images(seed: int, n: int):
    """수집기를 쓰지 않는 합성 이미지 (uint8, 128x128) 와 라벨. 세트마다 시드가 달라 서로 겹치지 않는다."""
    import numpy as np

    images = np.random.default_rng(seed).integers(0, 256, size=(n, 128, 128), dtype=np.uint8)
    return images, ["NG" if i % 3 == 0 else "OK" for i in range(n)]


@pytest.fixture(scope="session")
def trained_models(tmp_path_factory):
    """합성 이미지 24장으로 v1 하나를 학습해 둔다 (에폭 1, 아주 작음). 시험은 이 폴더의 models/v1 을 자기 폴더로 복사해서 쓴다.

    백엔드 시험은 모델의 정확도가 아니라 '불러오기, 판정, 저장' 연결을 확인하므로 이미지는 아무거나 괜찮다.
    수집기와 torch 는 이 fixture 를 쓰는 시험에서만 가져온다 (다른 백엔드 시험은 torch 없이 돈다).
    """
    from parts.predictor import train

    root = tmp_path_factory.mktemp("trained_models")
    config = root / "predictor.toml"
    config.write_text("threshold = 0.5\nepochs = 1\nbatch_size = 32\nlearning_rate = 0.001\nseed = 42\n"
                      'models_dir = "models"\n', encoding="utf-8")
    metrics = train(*_noise_images(1, 24), version="v1", eval_normal=_noise_images(2, 8), eval_drift=_noise_images(3, 8),
                    config_path=config, root=root)
    return SimpleNamespace(root=root, metrics=metrics, images=_noise_images(4, 6)[0])


# ---------- 3-D-2: 읽기 전용 조회 시험이 함께 쓰는 환경 (앱 1개, 시드는 한 번) ----------

@pytest.fixture(scope="session")
def read_env():
    """시드된 메모리 DB 와 앱·클라이언트 하나. GET 시험만 쓴다 (테스트 후 행 수가 그대로인지 마지막에 확인한다)."""
    from backend.database import session_scope
    from backend.repository import InspectionRepository, ModelVersionRepository, PredictionRepository
    from tests.backend.helpers import PREDICTION_SEED, bearer, seed_read_only_data

    settings = make_settings()
    app = create_app(settings)
    with session_scope(app.state.session_factory) as s:
        seed_read_only_data(s)

    def counts():
        with session_scope(app.state.session_factory) as s:
            return (InspectionRepository(s).count(), len(ModelVersionRepository(s).list()), PredictionRepository(s).count())

    expected = (12, 3, len(PREDICTION_SEED))
    assert counts() == expected
    yield SimpleNamespace(app=app, settings=settings, client=TestClient(app, raise_server_exceptions=False),
                          headers=bearer(settings, "operator"))
    try:
        after = counts()
        assert after == expected, f"읽기 전용 환경의 행 수가 바뀌었습니다: {expected} -> {after}"
    finally:
        app.state.engine.dispose()


@pytest.fixture(scope="session")
def empty_env():
    """행이 하나도 없는 앱 (빈 결과 형식 확인용, 읽기만 한다)."""
    from tests.backend.helpers import bearer

    settings = make_settings()
    app = create_app(settings)
    yield SimpleNamespace(app=app, settings=settings, client=TestClient(app, raise_server_exceptions=False),
                          headers=bearer(settings, "operator"))
    app.state.engine.dispose()
