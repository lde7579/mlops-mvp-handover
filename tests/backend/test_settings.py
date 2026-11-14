"""backend/settings.py: backend.toml 읽기, 설정 검증, 비밀키(환경 변수 > data/secret.key)."""
import json
import re

import pytest

from backend.security import Role, parse_hash
from backend.settings import (
    DEFAULT_CONFIG_PATH,
    MIN_ENV_SECRET_LENGTH,
    PROJECT_ROOT,
    SECRET_ENV,
    SettingsError,
    load_secret_key,
    load_settings,
)
from tests.backend.helpers import SECRET

GOOD_HASH = "pbkdf2_sha256$1000$c2FsdHNhbHRzYWx0c2FsdA$ZGlnZXN0ZGlnZXN0ZGlnZXN0ZGlnZXN0ZGlnZXN0ZGlnZXN0"


def write_toml(tmp_path, accounts=None, **over):
    values = {"db_path": "data/test.db", "token_minutes": 480}
    values.update(over)
    lines = []
    for key, v in values.items():
        if v is None:
            continue
        lines.append(f"{key} = " + ("true" if v is True else json.dumps(v, ensure_ascii=False) if isinstance(v, str) else repr(v)))
    accounts = [{"username": "operator", "role": "operator", "password_hash": GOOD_HASH}] if accounts is None else accounts
    for a in accounts:
        lines.append("[[accounts]]")
        lines += [f"{k} = {json.dumps(v, ensure_ascii=False)}" for k, v in a.items()]
    path = tmp_path / "backend.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_default_backend_toml_has_two_demo_accounts_and_no_secret():
    s = load_settings(secret_key=SECRET)
    assert [(a.username, a.role) for a in s.accounts] == [("operator", Role.OPERATOR), ("admin", Role.ADMIN)]
    assert s.token_minutes == 480
    assert s.db_path == str((PROJECT_ROOT / "data" / "mlops.db").resolve())
    assert all(parse_hash(a.password_hash)[0] == 600_000 for a in s.accounts)  # 해시만 있고 평문은 없다
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    assert "operator-demo" not in text and "admin-demo" not in text  # 비밀번호 평문은 없다
    assert SECRET not in text and not re.search(r"[0-9a-f]{64}", text)  # 비밀키도 없다


def test_valid_custom_settings(tmp_path):
    s = load_settings(write_toml(tmp_path), secret_key=SECRET, root=tmp_path)
    assert s.db_path == str((tmp_path / "data" / "test.db").resolve())
    assert s.account("operator").role is Role.OPERATOR and s.account("nobody") is None


@pytest.mark.parametrize("over", [
    {"token_minutes": 0}, {"token_minutes": -5}, {"token_minutes": "480"}, {"token_minutes": 4.5}, {"token_minutes": True},
    {"db_path": "/abs/db"}, {"db_path": "C:\\abs\\db"}, {"db_path": ""}, {"db_path": 5},
    {"unknown_key": 1}, {"db_path": None}, {"token_minutes": None},
])
def test_bad_top_level_values_are_rejected(tmp_path, over):
    with pytest.raises(SettingsError):
        load_settings(write_toml(tmp_path, **over), secret_key=SECRET, root=tmp_path)


@pytest.mark.parametrize("accounts", [
    [],
    [{"username": "a", "role": "operator"}],  # password_hash 없음
    [{"username": "a", "role": "boss", "password_hash": GOOD_HASH}],  # 모르는 역할
    [{"username": "a", "role": "operator", "password_hash": "plain-text-password"}],  # 해시 형식 아님
    [{"username": "a", "role": "operator", "password_hash": "md5$1$x$y"}],
    [{"username": "", "role": "operator", "password_hash": GOOD_HASH}],
    [{"username": "a", "role": "operator", "password_hash": GOOD_HASH},
     {"username": "a", "role": "admin", "password_hash": GOOD_HASH}],  # 이름 중복
    [{"username": "a", "role": "operator", "password_hash": GOOD_HASH, "extra": "x"}],
])
def test_bad_accounts_are_rejected(tmp_path, accounts):
    with pytest.raises(SettingsError):
        load_settings(write_toml(tmp_path, accounts=accounts), secret_key=SECRET, root=tmp_path)


# ---------- 비밀키 ----------

def test_env_secret_wins_and_no_file_is_created(tmp_path):
    key_file = tmp_path / "data" / "secret.key"
    key = "e" * 40
    assert load_secret_key({SECRET_ENV: key}, key_file) == key
    assert not key_file.exists() and not key_file.parent.exists()


@pytest.mark.parametrize("short", ["a", "x" * (MIN_ENV_SECRET_LENGTH - 1)])
def test_short_env_secret_is_rejected(tmp_path, short):
    with pytest.raises(SettingsError, match=SECRET_ENV):
        load_secret_key({SECRET_ENV: short}, tmp_path / "secret.key")


def test_empty_env_secret_counts_as_unset(tmp_path):
    key_file = tmp_path / "secret.key"
    assert len(load_secret_key({SECRET_ENV: "  "}, key_file)) == 64 and key_file.exists()


def test_secret_file_is_created_once_and_reused(tmp_path):
    key_file = tmp_path / "new_folder" / "secret.key"  # 폴더도 만든다
    first = load_secret_key({}, key_file)
    assert len(first) == 64 and all(c in "0123456789abcdef" for c in first)
    assert key_file.read_text(encoding="utf-8") == first + "\n"
    assert load_secret_key({}, key_file) == first  # 다시 시작해도 같은 값
    assert key_file.read_text(encoding="utf-8") == first + "\n"


@pytest.mark.parametrize("content", ["", "abc", "z" * 64, "A" * 64, "0" * 63, "0" * 65])
def test_broken_secret_file_stops_startup_and_is_not_overwritten(tmp_path, content):
    key_file = tmp_path / "secret.key"
    key_file.write_text(content, encoding="utf-8")
    with pytest.raises(SettingsError, match="지우고 다시"):
        load_secret_key({}, key_file)
    assert key_file.read_text(encoding="utf-8") == content


def test_load_settings_uses_env_then_file_when_no_key_is_injected(tmp_path):
    key = "f" * 48
    s = load_settings(write_toml(tmp_path), env={SECRET_ENV: key}, root=tmp_path, secret_file=tmp_path / "k")
    assert s.secret_key == key
    s2 = load_settings(write_toml(tmp_path), env={}, root=tmp_path, secret_file=tmp_path / "k")
    assert s2.secret_key == (tmp_path / "k").read_text(encoding="utf-8").strip()
