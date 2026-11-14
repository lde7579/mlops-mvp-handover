"""폴더 틀 확인용: 구조가 갖춰져 있고 import가 되는지만 본다 (spike는 import하지 않는다)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_parts_and_common_importable():
    import common.preprocess  # noqa: F401
    import parts.collector  # noqa: F401
    import parts.predictor  # noqa: F401


def test_reference_docs_exist():
    for name in ("requirements.md", "data-contract.md", "api-v1.md", "screens.md", "e2e-scenario.md"):
        assert (ROOT / "docs" / name).is_file(), name


def test_test_folders_exist():
    assert (ROOT / "tests" / "contract").is_dir()
    assert (ROOT / "tests" / "e2e").is_dir()
