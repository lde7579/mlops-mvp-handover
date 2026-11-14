"""검사 이미지 파일 경로 확인. DB 의 image_path 를 믿지 않고, 허용된 폴더 안의 진짜 PNG 파일일 때만 경로를 돌려준다.

허용: 프로젝트 루트의 data/ 아래 .png 파일 (data/pool/…, data/dataset/<split>/… 등).
거부(모두 None): 빈 값, 절대경로·드라이브 문자, '..', 'data/' 밖, .png 가 아닌 파일, 없는 파일,
                  경로 중간에 심볼릭 링크·정션(폴더 연결)이 있는 경우.
링크 판별: 실제 경로(resolve)와 글자 그대로 이어 붙인 경로가 다르면 어딘가에 링크가 있는 것이다 (Python 3.11 은
Windows 정션을 is_symlink 로 알아내지 못하므로 이 방법을 쓴다). 거부 사유는 호출한 쪽에 알리지 않고 로그에만 남긴다.
"""
import logging
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Optional

log = logging.getLogger("backend")

ALLOWED_DIR = "data"
ALLOWED_SUFFIX = ".png"


def _same(a: Path, b: Path) -> bool:
    return os.path.normcase(str(a)) == os.path.normcase(str(b))


def resolve_image(root, image_path) -> Optional[Path]:
    """image_path(프로젝트 루트 기준 상대경로)가 가리키는 실제 파일 경로. 허용되지 않으면 None."""
    if not isinstance(image_path, str) or not image_path or "\x00" in image_path:
        return _deny(image_path, "빈 값이거나 문자열이 아님")
    win, posix = PureWindowsPath(image_path), PurePosixPath(image_path)
    if win.anchor or win.drive or posix.anchor:
        return _deny(image_path, "절대경로 또는 드라이브 지정")
    parts = win.parts
    if any(p == ".." for p in parts) or any(":" in p for p in parts):
        return _deny(image_path, "'..' 또는 ':' 포함")
    if not parts or parts[0] != ALLOWED_DIR or len(parts) < 2 or not parts[-1].lower().endswith(ALLOWED_SUFFIX):
        return _deny(image_path, "data/ 아래 .png 가 아님")

    base = Path(root).resolve()
    lexical = base.joinpath(*parts)
    try:
        real = lexical.resolve(strict=True)
    except (OSError, RuntimeError):
        return _deny(image_path, "파일이 없거나 열 수 없음")
    if not _same(real, lexical):
        return _deny(image_path, "경로 중간에 링크가 있음")
    if not real.is_file():
        return _deny(image_path, "파일이 아님")
    return real


def _deny(image_path, reason: str) -> None:
    log.warning("이미지 경로 거부 (%s): %r", reason, image_path)
    return None
