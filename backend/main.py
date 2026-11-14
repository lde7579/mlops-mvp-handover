"""FastAPI 앱 생성. 파일을 불러오는 것만으로는 DB 도 비밀키도 만들지 않는다 (앱을 만들 때 만든다).

실행: python -m uvicorn backend.main:create_app --factory --port 8000
"""
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from backend.database import create_db_engine, init_db, make_session_factory
from backend.deployment import DeployService
from backend.errors import install_error_handlers, install_openapi
from backend.generation import GenerationService
from backend.inference import InferenceService
from backend.monitoring import MonitoringService
from backend.retrain import RetrainService
from backend.routers import (
    approvals,
    audit_logs,
    auth,
    deployments,
    health,
    inspections,
    models,
    monitoring,
    predict,
    predictions,
    simulator,
    training,
)
from backend.settings import Settings, load_settings
from backend.simulator_runner import build_runner

log = logging.getLogger("backend")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = load_settings() if settings is None else settings
    engine = create_db_engine(settings.db_path)
    init_db(engine)

    session_factory = make_session_factory(engine)
    inference = InferenceService(settings)
    retrain = RetrainService(settings, session_factory)
    deploy = DeployService(session_factory, inference)
    generation = GenerationService(settings)  # 수집기는 첫 생성 요청 때 만든다 (여기서는 파일을 만들지 않음)
    runner = build_runner(settings, session_factory, generation, inference)  # 자동 생성은 POST /simulator/start 때 스레드를 시작한다

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:  # 서버가 켜질 때만: 학습 중에 꺼져서 queued·running 으로 남은 기록을 failed 로 정리한다 (실패해도 서버는 뜬다)
            retrain.mark_interrupted()
        except Exception:
            log.exception("시작할 때 남은 재학습 기록을 정리하지 못했습니다")
        inference.preload(app.state.session_factory)  # 서버가 켜질 때만. 실패해도 서버는 뜬다
        yield
        app.state.simulator.shutdown()  # 자동 생성 스레드를 먼저 멈춘 뒤 DB 연결을 닫는다 (스레드가 닫힌 DB 에 쓰지 않게)
        engine.dispose()

    app = FastAPI(title="MLOps MVP 백엔드", version="1", lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.retrain = retrain                          # 학습은 POST /training/runs 때 백그라운드 스레드로 시작한다
    app.state.generation = generation
    app.state.monitoring = MonitoringService(settings.root)  # 성능 추이·입력 분포 (기준 통계는 처음 요청 때 계산)
    app.state.simulator = runner                         # 자동 생성·판정 (프로세스 하나만 지원, 드리프트 주입 상태는 메모리)
    app.state.deploy = deploy                            # 승인·거절·롤백 (모델은 먼저 읽어 두고, 커밋한 뒤에 교체한다)
    app.state.inference = inference                    # 모델은 서버가 켜질 때(lifespan)나 첫 판정 때 불러온다
    install_error_handlers(app)
    install_openapi(app)
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(simulator.router)
    app.include_router(inspections.router)
    app.include_router(predict.router)
    app.include_router(predictions.router)
    app.include_router(models.router)
    app.include_router(monitoring.router)
    app.include_router(training.router)
    app.include_router(approvals.router)
    app.include_router(deployments.router)
    app.include_router(audit_logs.router)
    return app
