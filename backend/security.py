"""보안 도구: 비밀번호 해시(PBKDF2-SHA256, 표준 라이브러리), 로그인 토큰(HS256 JWT, PyJWT), 역할.

- 비밀번호는 해시만 저장한다: pbkdf2_sha256$반복횟수$salt$해시 (평문은 어디에도 두지 않는다).
- 토큰에는 sub(계정 이름), iat(발급 시각), exp(만료 시각)만 담는다. 역할은 담지 않고 계정 설정에서 찾는다.
- 해시 만들기: python -m backend.security  (비밀번호를 입력하면 backend.toml 에 넣을 해시를 출력한다)
"""
import base64
import binascii
import hashlib
import hmac
import re
import secrets
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Optional

import jwt

HASH_SCHEME = "pbkdf2_sha256"
PBKDF2_ITERATIONS = 600_000  # 이 PC 에서 로그인 한 번에 약 0.3초
ALGORITHM = "HS256"
_B64_TEXT = re.compile(r"[A-Za-z0-9_-]+")


class Role(StrEnum):
    OPERATOR = "operator"  # 운영자
    ADMIN = "admin"        # 관리자 (운영자 권한을 모두 포함)


ROLE_RANK = {Role.OPERATOR: 1, Role.ADMIN: 2}


class TokenError(Exception):
    """토큰이 없거나 올바르지 않거나 만료되었다."""


# ---------- 비밀번호 ----------

def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> bytes:
    # urlsafe_b64decode 는 알파벳 밖 문자를 조용히 버리므로, 먼저 형식을 엄격하게 확인한다
    if not _B64_TEXT.fullmatch(text):
        raise ValueError(text)
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS, salt: Optional[bytes] = None) -> str:
    salt = secrets.token_bytes(16) if salt is None else salt
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{HASH_SCHEME}${iterations}${_b64e(salt)}${_b64e(digest)}"


def parse_hash(stored: str) -> tuple:
    """(반복횟수, salt, 해시). 형식이 틀리면 ValueError."""
    try:
        scheme, iterations, salt, digest = stored.split("$")
        if scheme != HASH_SCHEME:
            raise ValueError(scheme)
        count = int(iterations)
        if count < 1:
            raise ValueError(iterations)
        return count, _b64d(salt), _b64d(digest)  # salt 와 해시는 비어 있으면 안 된다 (정규식이 1자 이상을 요구)
    except (ValueError, TypeError, AttributeError, binascii.Error) as e:
        raise ValueError(f"비밀번호 해시 형식이 올바르지 않습니다 ({HASH_SCHEME}$횟수$salt$해시)") from e


def verify_password(password: str, stored: str) -> bool:
    """맞으면 True. 해시 형식이 깨졌으면 False (시간 차이를 줄이려고 같은 방식으로 비교한다)."""
    try:
        iterations, salt, expected = parse_hash(stored)
    except ValueError:
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


# ---------- 토큰 ----------

def create_token(username: str, secret: str, minutes: int, now: Optional[datetime] = None) -> str:
    now = datetime.now(timezone.utc) if now is None else now
    payload = {
        "sub": username,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=minutes)).timestamp()),
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def decode_token(token: str, secret: str) -> str:
    """검증에 성공하면 계정 이름(sub). 서명·만료·알고리즘·필수 항목(exp, sub, iat)을 모두 확인한다."""
    try:
        payload = jwt.decode(token, secret, algorithms=[ALGORITHM], options={"require": ["exp", "sub", "iat"]})
    except jwt.PyJWTError as e:
        raise TokenError(type(e).__name__) from e
    username = payload.get("sub")
    if not isinstance(username, str) or not username:
        raise TokenError("sub")
    return username


if __name__ == "__main__":  # 계정 비밀번호 해시 만들기
    from getpass import getpass

    print(hash_password(getpass("비밀번호: ")))
