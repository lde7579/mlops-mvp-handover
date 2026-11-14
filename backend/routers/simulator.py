"""시뮬레이터: 검사 1건 수동 생성(generate), 자동 생성 시작·중지(start, stop), 드리프트 주입(drift), 상태 조회(status).

generate·start·stop·drift 는 운영자 이상, status 는 로그인한 누구나 (api-v1.md 3-2, A7·A23).
"""
from typing import Annotated

from fastapi import APIRouter, Body, Depends

from backend.api_models import (
    DriftRequest,
    ErrorResponse,
    GenerateRequest,
    InspectionOut,
    SimulatorStatusOut,
    StartRequest,
)
from backend.deps import CurrentUser, GenerationDep, SessionDep, SimulatorDep, User, require_role
from backend.errors import ApiError
from backend.security import Role
from backend.simulator_runner import SimulatorBusyError

router = APIRouter(prefix="/simulator", tags=["simulator"])

_ERRORS = {code: {"model": ErrorResponse} for code in (400, 401, 403)}


@router.post("/generate", status_code=201, response_model=InspectionOut, responses=_ERRORS)
def generate(session: SessionDep, generation: GenerationDep,
             _: Annotated[User, Depends(require_role(Role.OPERATOR))],
             body: Annotated[GenerateRequest, Body()] = GenerateRequest()) -> InspectionOut:
    """검사 1건을 만들어 data/pool 에 저장하고 DB 에 등록한다. 요청 본문은 없어도 된다 (scenario 기본값 normal)."""
    return InspectionOut.of(generation.generate(session, body.scenario.value))


@router.post("/start", response_model=SimulatorStatusOut, responses={**_ERRORS, 409: {"model": ErrorResponse}})
def start(simulator: SimulatorDep, _: Annotated[User, Depends(require_role(Role.OPERATOR))],
          body: Annotated[StartRequest, Body()] = StartRequest()) -> SimulatorStatusOut:
    """자동 생성을 시작한다. 이미 실행 중이면 409. 운영 모델이 없어도 시작되고 그때는 판정만 건너뛴다."""
    try:
        return SimulatorStatusOut.of(simulator.start(body.defect_rate, body.interval_seconds))
    except SimulatorBusyError as e:
        raise ApiError(409, str(e)) from None


@router.post("/stop", response_model=SimulatorStatusOut, responses=_ERRORS)
def stop(simulator: SimulatorDep, _: Annotated[User, Depends(require_role(Role.OPERATOR))]) -> SimulatorStatusOut:
    """자동 생성을 멈춘다 (스레드가 끝난 뒤 응답). 실행 중이 아니어도 200."""
    return SimulatorStatusOut.of(simulator.stop())


@router.post("/drift", response_model=SimulatorStatusOut, responses=_ERRORS)
def drift(simulator: SimulatorDep, _: Annotated[User, Depends(require_role(Role.OPERATOR))],
          body: Annotated[DriftRequest, Body()]) -> SimulatorStatusOut:
    """드리프트 주입을 켜거나 끈다. 켜면 drift_type 필수, 끄면 drift_type 을 보내지 않는다."""
    if body.enabled and body.drift_type is None:
        raise ApiError(400, "enabled 가 true 이면 drift_type 이 필요합니다 (dark, noise, shift)")
    if not body.enabled and body.drift_type is not None:
        raise ApiError(400, "enabled 가 false 이면 drift_type 을 보낼 수 없습니다")
    return SimulatorStatusOut.of(simulator.set_drift(body.enabled, body.drift_type))


@router.get("/status", response_model=SimulatorStatusOut, responses={401: {"model": ErrorResponse}})
def status(simulator: SimulatorDep, _: CurrentUser) -> SimulatorStatusOut:
    """자동 생성 상태 (실행 여부, 설정, 드리프트 주입, 만든·판정한·건너뛴 건수, 마지막 오류)."""
    return SimulatorStatusOut.of(simulator.status())
