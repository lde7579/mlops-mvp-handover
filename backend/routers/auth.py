"""POST /auth/login: 계정·비밀번호로 토큰 발급 (공개)."""
from typing import Optional

from fastapi import APIRouter

from backend.api_models import ErrorResponse, LoginRequest, LoginResponse
from backend.deps import SettingsDep
from backend.errors import BEARER_HEADER, ApiError
from backend.security import create_token, verify_password
from backend.settings import Account, Settings

router = APIRouter(prefix="/auth", tags=["auth"])

LOGIN_FAILED = "계정 또는 비밀번호가 올바르지 않습니다"  # 없는 계정과 틀린 비밀번호에 같은 메시지를 쓴다


def authenticate(settings: Settings, username: str, password: str) -> Optional[Account]:
    account = settings.account(username)
    if account is None:
        # 없는 계정에도 해시 비교를 한 번 해서, 응답 시간으로 계정이 있는지 알아내지 못하게 한다
        verify_password(password, settings.accounts[0].password_hash)
        return None
    return account if verify_password(password, account.password_hash) else None


@router.post("/login", response_model=LoginResponse, responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}})
def login(body: LoginRequest, settings: SettingsDep) -> LoginResponse:
    account = authenticate(settings, body.username, body.password)
    if account is None:
        raise ApiError(401, LOGIN_FAILED, BEARER_HEADER)
    token = create_token(account.username, settings.secret_key, settings.token_minutes)
    return LoginResponse(access_token=token, expires_in=settings.token_minutes * 60, role=account.role,
                         username=account.username)
