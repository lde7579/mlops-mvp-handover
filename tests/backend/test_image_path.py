"""backend/images.py: 이미지 경로 확인 (data/ 아래 진짜 PNG 파일만 허용)."""
import logging

import pytest

from backend.images import resolve_image
from tests.backend.helpers import make_dir_link, png_bytes, write_file


@pytest.fixture
def root(tmp_path):
    write_file(tmp_path, "data/pool/INS-000001.png", png_bytes())
    write_file(tmp_path, "data/dataset/train/INS-000002.png", png_bytes())
    write_file(tmp_path, "data/mlops.db")
    write_file(tmp_path, "data/secret.key")
    write_file(tmp_path, "data/pool/note.txt")
    write_file(tmp_path, "outside.png", png_bytes())
    write_file(tmp_path, "parts/collector/collector.toml")
    return tmp_path


@pytest.mark.parametrize("rel", ["data/pool/INS-000001.png", "data/dataset/train/INS-000002.png",
                                 "data\\pool\\INS-000001.png", "data/pool/./INS-000001.png"])
def test_png_under_data_is_allowed(root, rel):
    found = resolve_image(root, rel)
    assert found is not None and found.is_file() and found.suffix == ".png"


def test_suffix_check_ignores_case(root):
    write_file(root, "data/pool/UPPER.PNG", png_bytes())
    assert resolve_image(root, "data/pool/UPPER.PNG") is not None


@pytest.mark.parametrize("rel", [
    "", None, 5, b"data/pool/INS-000001.png", "data/pool/\x00.png",
    "..", "../outside.png", "data/../outside.png", "data\\..\\outside.png", "data/pool/../../outside.png",
    "data/pool/..\\..\\outside.png",
    "/data/pool/INS-000001.png", "C:/data/pool/INS-000001.png", "C:data/pool/INS-000001.png",
    "\\\\server\\share\\a.png", "\\data\\pool\\INS-000001.png",
    "data/pool/INS-000001.png:stream", "data/pool/INS-000001.png::$DATA",
    "data", "data/", "data/pool", "data/png",
    "outside.png", "parts/collector/collector.toml", "Data/pool/INS-000001.png",
    "data/mlops.db", "data/secret.key", "data/pool/note.txt", "data/pool/INS-000001.png.txt",
    "data/pool/missing.png",
])
def test_everything_else_is_denied(root, rel):
    assert resolve_image(root, rel) is None


def test_absolute_path_of_a_real_file_is_denied(root):
    assert resolve_image(root, str(root / "data" / "pool" / "INS-000001.png")) is None


def test_directory_named_like_png_is_denied(root):
    (root / "data" / "pool" / "dir.png").mkdir()
    assert resolve_image(root, "data/pool/dir.png") is None


def test_junction_to_outside_folder_is_denied(root, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    write_file(outside, "secret.png", png_bytes())
    make_dir_link(root / "data" / "link", outside)
    assert resolve_image(root, "data/link/secret.png") is None


def test_junction_to_another_folder_inside_data_is_also_denied(root):
    make_dir_link(root / "data" / "alias", root / "data" / "pool")
    assert resolve_image(root, "data/pool/INS-000001.png") is not None
    assert resolve_image(root, "data/alias/INS-000001.png") is None  # 링크는 안쪽을 가리켜도 허용하지 않는다


def test_data_folder_itself_being_a_link_is_denied(tmp_path, tmp_path_factory):
    real = tmp_path_factory.mktemp("realdata")
    write_file(real, "pool/INS-000001.png", png_bytes())
    make_dir_link(tmp_path / "data", real)
    assert resolve_image(tmp_path, "data/pool/INS-000001.png") is None


def test_denied_path_is_logged_with_the_reason_and_not_returned(root, caplog):
    with caplog.at_level(logging.WARNING, logger="backend"):
        assert resolve_image(root, "../outside.png") is None
    assert "이미지 경로 거부" in caplog.text and "'..'" in caplog.text
