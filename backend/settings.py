"""백엔드 설정: backend.toml(DB 경로, 토큰 시간, 계정)과 비밀키.

비밀키는 코드와 저장소 파일에 없다. 우선순위:
  1. 환경 변수 MLOPS_SECRET_KEY (32자 이상)
  2. data/secret.key 파일 (16진수 64자). 없으면 처음 시작할 때 무작위로 만들어 저장한다.
     깨졌거나 짧으면 시작을 거부하고 덮어쓰지 않는다 (덮어쓰면 이미 발급된 토큰이 조용히 무효화된다).
data/ 는 git 과 조원 전달 압축에서 제외된다.
"""
import os
import re
import secrets
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Mapping, Optional

from backend.security import Role, parse_hash

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("backend.toml")
SECRET_ENV = "MLOPS_SECRET_KEY"
SECRET_FILE = PROJECT_ROOT / "data" / "secret.key"
MIN_ENV_SECRET_LENGTH = 32
_HEX64 = re.compile(r"[0-9a-f]{64}")
_KEYS = {"db_path", "token_minutes", "accounts", "collector_config"}
_OPTIONAL_KEYS = {"collector_config"}  # 없으면 기본값 (수집기 기본 설정 파일)
DEFAULT_COLLECTOR_CONFIG = PROJECT_ROOT / "parts" / "collector" / "collector.toml"
_ACCOUNT_KEYS = {"username", "role", "password_hash"}


class SettingsError(ValueError):
    """설정 파일이나 비밀키가 규칙에 맞지 않음."""


@dataclass(frozen=True)
class Account:
    username: str
    role: Role
    password_hash: str


@dataclass(frozen=True)
class Settings:
    db_path: str              # 파일 경로 또는 ":memory:"
    token_minutes: int
    accounts: tuple
    secret_key: str
    root: str = str(PROJECT_ROOT)                              # 프로젝트 루트 (image_path 와 수집기 data 폴더의 기준)
    collector_config: str = str(DEFAULT_COLLECTOR_CONFIG)      # 수집기 설정 파일 경로

    def account(self, username: str) -> Optional[Account]:
        for a in self.accounts:
            if a.username == username:
                return a
        return None


def load_secret_key(env: Optional[Mapping] = None, secret_file=SECRET_FILE) -> str:
    env = os.environ if env is None else env
    value = (env.get(SECRET_ENV) or "").strip()
    if value:
        if len(value) < MIN_ENV_SECRET_LENGTH:
            raise SettingsError(f"{SECRET_ENV}: {MIN_ENV_SECRET_LENGTH}자 이상이어야 합니다")
        return value

    path = Path(secret_file)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "x", encoding="utf-8") as f:  # 'x': 이미 있으면 덮어쓰지 않는다
                f.write(secrets.token_hex(32) + "\n")
        except FileExistsError:
            pass  # 동시에 시작한 다른 프로세스가 먼저 만들었다 -> 그 값을 읽는다
    text = path.read_text(encoding="utf-8").strip()
    if not _HEX64.fullmatch(text):
        raise SettingsError(
            f"{path}: 비밀키 파일이 깨졌거나 짧습니다 (16진수 64자여야 함). 파일을 지우고 다시 시작하세요 "
            "(지우면 이미 발급된 토큰은 모두 무효가 됩니다)")
    return text


def _is_relative(path_text: str) -> bool:
    return bool(path_text) and not (PurePosixPath(path_text).anchor or PureWindowsPath(path_text).anchor)


def load_settings(config_path=None, *, env: Optional[Mapping] = None, root=PROJECT_ROOT,
                  secret_file=SECRET_FILE, secret_key: Optional[str] = None) -> Settings:
    """설정을 읽는다. secret_key 를 주면 환경 변수·파일을 보지 않는다(테스트용)."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    missing, unknown = sorted(_KEYS - _OPTIONAL_KEYS - raw.keys()), sorted(raw.keys() - _KEYS)
    if missing:
        raise SettingsError(f"설정에 없는 항목: {', '.join(missing)}")
    if unknown:
        raise SettingsError(f"알 수 없는 설정 항목: {', '.join(unknown)}")

    db_path = raw["db_path"]
    if not isinstance(db_path, str) or not _is_relative(db_path):
        raise SettingsError(f"db_path: 프로젝트 루트 기준 상대경로여야 합니다 (받은 값: {db_path!r})")
    collector_config = raw.get("collector_config", "parts/collector/collector.toml")
    if not isinstance(collector_config, str) or not _is_relative(collector_config):
        raise SettingsError(f"collector_config: 프로젝트 루트 기준 상대경로여야 합니다 (받은 값: {collector_config!r})")
    minutes = raw["token_minutes"]
    if isinstance(minutes, bool) or not isinstance(minutes, int) or minutes < 1:
        raise SettingsError(f"token_minutes: 1 이상의 정수여야 합니다 (받은 값: {minutes!r})")

    entries = raw["accounts"]
    if not isinstance(entries, list) or not entries:
        raise SettingsError("accounts: 계정이 1개 이상 있어야 합니다")
    accounts = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != _ACCOUNT_KEYS:
            raise SettingsError(f"accounts: 각 계정은 {sorted(_ACCOUNT_KEYS)} 항목이 필요합니다 (받은 값: {entry!r})")
        username, role, stored = entry["username"], entry["role"], entry["password_hash"]
        if not isinstance(username, str) or not username:
            raise SettingsError(f"accounts: username 이 올바르지 않습니다 ({username!r})")
        try:
            role = Role(role)
        except ValueError:
            raise SettingsError(f"accounts[{username}]: 모르는 역할입니다 ({role!r}). 허용값: operator, admin") from None
        try:
            parse_hash(stored)
        except ValueError as e:
            raise SettingsError(f"accounts[{username}]: {e}") from None
        accounts.append(Account(username, role, stored))
    names = [a.username for a in accounts]
    if len(set(names)) != len(names):
        raise SettingsError(f"accounts: 계정 이름이 겹칩니다 ({names})")

    key = secret_key if secret_key is not None else load_secret_key(env, secret_file)
    root_path = Path(root).resolve()
    return Settings(db_path=str((root_path / db_path).resolve()), token_minutes=minutes,
                    accounts=tuple(accounts), secret_key=key, root=str(root_path),
                    collector_config=str((root_path / collector_config).resolve()))
