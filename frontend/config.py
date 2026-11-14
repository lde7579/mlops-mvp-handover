"""화면 설정. 설정은 이 파일 한 곳에서만 정한다.

- 서버 주소는 환경 변수 MLOPS_API_URL (기본 http://127.0.0.1:8000). 호출할 때마다 읽는다.
- 갱신 주기는 3~5초 (docs/api-v1.md 1장, screens.md 1장).
- 연결 제한 시간은 짧게 둔다: 서버가 꺼져 있어도 화면이 오래 멈추지 않게 하기 위해서다.
"""
import os

DEFAULT_API_URL = "http://127.0.0.1:8000"
API_URL_ENV = "MLOPS_API_URL"
REFRESH_SECONDS = 4
CONNECT_TIMEOUT_SECONDS = 1.0
READ_TIMEOUT_SECONDS = 3.0
LOGIN_ACCOUNTS = ("operator", "admin")


def api_url() -> str:
    return os.environ.get(API_URL_ENV, DEFAULT_API_URL).strip().rstrip("/") or DEFAULT_API_URL


def refresh_seconds() -> int:
    return REFRESH_SECONDS
