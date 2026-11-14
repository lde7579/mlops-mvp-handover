"""엔드포인트가 재사용하는 의존성: 설정, DB 세션, 로그인한 사용자, 역할 확인.

3-C 이후 엔드포인트는 이렇게 붙이면 된다:
    def handler(session: SessionDep, user: CurrentUser): ...                      # 로그인한 누구나 (조회)
    def handler(session: SessionDep, user: Annotated[User, Depends(require_role(Role.OPERATOR))]): ...  # 운영자 이상
    def handler(session: SessionDep, user: Annotated[User, Depends(require_role(Role.ADMIN))]): ...     # 관리자만
"""
from dataclasses import dataclass
from typing import Annotated, Optional

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.database import session_scope
from backend.deployment import DeployService
from backend.errors import BEARER_HEADER, ApiError
from backend.generation import GenerationService
from backend.inference import InferenceService
from backend.monitoring import MonitoringService
from backend.retrain import RetrainService
from backend.security import ROLE_RANK, Role, TokenError, decode_token
from backend.settings import Settings
from backend.simulator_runner import SimulatorRunner


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_generation(request: Request) -> GenerationService:
    return request.app.state.generation


GenerationDep = Annotated[GenerationService, Depends(get_generation)]


def get_monitoring(request: Request) -> MonitoringService:
    return request.app.state.monitoring


MonitoringDep = Annotated[MonitoringService, Depends(get_monitoring)]


def get_simulator(request: Request) -> SimulatorRunner:
    return request.app.state.simulator


SimulatorDep = Annotated[SimulatorRunner, Depends(get_simulator)]


def get_inference(request: Request) -> InferenceService:
    return request.app.state.inference


InferenceDep = Annotated[InferenceService, Depends(get_inference)]


def get_retrain(request: Request) -> RetrainService:
    return request.app.state.retrain


RetrainDep = Annotated[RetrainService, Depends(get_retrain)]


def get_deploy(request: Request) -> DeployService:
    return request.app.state.deploy


DeployDep = Annotated[DeployService, Depends(get_deploy)]


def get_session(request: Request):
    """요청마다 세션 하나. 핸들러가 정상 종료하면 커밋, 예외면 롤백, 끝나면 닫는다."""
    with session_scope(request.app.state.session_factory) as session:
        yield session


# scope="function": 종료 코드(커밋)가 응답을 보내기 *전에* 실행된다. 기본값(request)은 응답을 보낸 뒤에 커밋해서,
# 쓰기 응답을 받자마자 다른 요청이 읽으면 데이터가 안 보일 수 있다 (tests/backend/test_session_dependency.py 가 확인).
SessionDep = Annotated[Session, Depends(get_session, scope="function")]

_bearer = HTTPBearer(auto_error=False, description="POST /auth/login 으로 받은 access_token")
_LABELS = {Role.OPERATOR: "운영자 이상", Role.ADMIN: "관리자"}


@dataclass(frozen=True)
class User:
    username: str
    role: Role


def get_current_user(
    settings: SettingsDep,
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(_bearer)],
) -> User:
    """Authorization: Bearer <토큰> 을 확인한다. 없거나 틀리거나 만료되면 401.

    역할은 토큰이 아니라 현재 계정 설정에서 찾는다 (설정에서 빠진 계정의 토큰은 거부된다).
    """
    if credentials is None:
        raise ApiError(401, "인증이 필요합니다", BEARER_HEADER)
    try:
        username = decode_token(credentials.credentials, settings.secret_key)
    except TokenError:
        raise ApiError(401, "토큰이 올바르지 않거나 만료되었습니다", BEARER_HEADER) from None
    account = settings.account(username)
    if account is None:
        raise ApiError(401, "토큰이 올바르지 않거나 만료되었습니다", BEARER_HEADER)
    return User(account.username, account.role)


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_role(minimum: Role):
    """minimum 이상의 역할만 통과시키는 의존성을 만든다 (관리자는 운영자 권한을 모두 가진다). 부족하면 403."""

    def dependency(user: CurrentUser) -> User:
        if ROLE_RANK[user.role] < ROLE_RANK[minimum]:
            raise ApiError(403, f"이 작업은 {_LABELS[minimum]}만 할 수 있습니다")
        return user

    return dependency
