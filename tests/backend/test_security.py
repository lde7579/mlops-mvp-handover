"""backend/security.py: 비밀번호 해시와 토큰(HS256 JWT)."""
import base64
import json
from datetime import datetime, timedelta, timezone

import jwt
import pytest

from backend.security import (
    ROLE_RANK,
    Role,
    TokenError,
    create_token,
    decode_token,
    hash_password,
    parse_hash,
    verify_password,
)
from tests.backend.helpers import OTHER_SECRET, SECRET

NOW = datetime(2026, 10, 18, 9, 0, 0, tzinfo=timezone.utc)
FAST = 1000


def b64(data: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()


def claims(**over):
    values = {"sub": "operator", "iat": int(NOW.timestamp()), "exp": int((NOW + timedelta(hours=8)).timestamp())}
    values.update(over)
    return {k: v for k, v in values.items() if v is not None}


# ---------- 비밀번호 ----------

def test_hash_has_documented_format_and_never_contains_the_password():
    h = hash_password("pw-secret", iterations=FAST)
    scheme, iterations, salt, digest = h.split("$")
    assert (scheme, iterations) == ("pbkdf2_sha256", "1000") and salt and digest
    assert "pw-secret" not in h
    assert parse_hash(h)[0] == FAST


def test_same_password_gives_different_hashes_because_of_salt():
    assert hash_password("pw", iterations=FAST) != hash_password("pw", iterations=FAST)


def test_correct_and_wrong_passwords():
    h = hash_password("올바른-비밀번호", iterations=FAST)
    assert verify_password("올바른-비밀번호", h)
    assert not verify_password("틀린-비밀번호", h)
    assert not verify_password("", h)
    assert not verify_password("올바른-비밀번호 ", h)


@pytest.mark.parametrize("stored", ["", "plain", "a$b$c", "pbkdf2_sha256$x$c2FsdA$ZGln", "pbkdf2_sha256$0$c2FsdA$ZGln",
                                    "md5$1000$c2FsdA$ZGln", "pbkdf2_sha256$1000$***$ZGln", None, 5])
def test_broken_hash_is_rejected(stored):
    with pytest.raises(ValueError):
        parse_hash(stored)
    assert verify_password("pw", stored) is False  # 깨진 해시는 오류가 아니라 거부


def test_default_iterations_are_the_documented_600000():
    assert parse_hash(hash_password("pw"))[0] == 600_000


def test_roles_admin_includes_operator():
    assert ROLE_RANK[Role.ADMIN] > ROLE_RANK[Role.OPERATOR]
    assert {r.value for r in Role} == {"operator", "admin"}


# ---------- 토큰 ----------

def test_token_round_trip_gives_the_username():
    assert decode_token(create_token("admin", SECRET, 480), SECRET) == "admin"


def test_token_carries_only_sub_iat_exp_and_lifetime_matches_setting():
    payload = jwt.decode(create_token("operator", SECRET, 480, now=NOW), options={"verify_signature": False})
    assert set(payload) == {"sub", "iat", "exp"}  # 역할은 담지 않는다
    assert payload["sub"] == "operator" and payload["exp"] - payload["iat"] == 480 * 60


def test_expired_token_is_rejected():
    token = create_token("operator", SECRET, 480, now=datetime.now(timezone.utc) - timedelta(hours=9))
    with pytest.raises(TokenError):
        decode_token(token, SECRET)


def test_token_signed_with_another_key_is_rejected():
    with pytest.raises(TokenError):
        decode_token(create_token("operator", OTHER_SECRET, 480), SECRET)


def test_tampered_payload_is_rejected():
    head, payload, sig = create_token("operator", SECRET, 480).split(".")
    forged = b64({**claims(), "sub": "admin"})
    with pytest.raises(TokenError):
        decode_token(f"{head}.{forged}.{sig}", SECRET)


def test_alg_none_token_is_rejected():
    unsigned = f"{b64({'alg': 'none', 'typ': 'JWT'})}.{b64(claims(exp=9999999999))}."
    with pytest.raises(TokenError):
        decode_token(unsigned, SECRET)


def test_other_algorithm_is_rejected():
    token = jwt.encode(claims(exp=9999999999), SECRET, algorithm="HS512")
    with pytest.raises(TokenError):
        decode_token(token, SECRET)


@pytest.mark.parametrize("missing", ["exp", "sub", "iat"])
def test_token_without_required_claim_is_rejected(missing):
    payload = {**claims(), "exp": 9999999999}
    del payload[missing]  # 필수 항목 하나를 뺀다
    token = jwt.encode(payload, SECRET, algorithm="HS256")
    with pytest.raises(TokenError):
        decode_token(token, SECRET)


@pytest.mark.parametrize("sub", ["", 123, None])
def test_token_with_bad_subject_is_rejected(sub):
    payload = {**claims(), "exp": 9999999999}
    if sub is None:
        del payload["sub"]
    else:
        payload["sub"] = sub
    token = jwt.encode(payload, SECRET, algorithm="HS256")
    with pytest.raises(TokenError):
        decode_token(token, SECRET)


@pytest.mark.parametrize("garbage", ["", "not-a-token", "a.b.c", "....", "Bearer abc"])
def test_garbled_token_is_rejected(garbage):
    with pytest.raises(TokenError):
        decode_token(garbage, SECRET)
