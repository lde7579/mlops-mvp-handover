"""GET /health: 백엔드·DB·운영 모델 상태 (공개, 인증 없음)."""
import logging

from fastapi import APIRouter
from sqlalchemy import text

from backend.api_models import HealthResponse, ModelStatus
from backend.deps import InferenceDep, SessionDep
from backend.repository import ModelVersionRepository

router = APIRouter(tags=["system"])
log = logging.getLogger("backend")


@router.get("/health", response_model=HealthResponse)
def health(session: SessionDep, inference: InferenceDep) -> HealthResponse:
    """백엔드가 응답하면 항상 200. DB 에 문제가 있으면 status 가 degraded 이고 오류 상세는 노출하지 않는다."""
    production_version = None
    try:
        session.execute(text("SELECT 1"))
        record = ModelVersionRepository(session).get_production()
        production_version = record.version if record else None
        database = "ok"
    except Exception:  # 연결 실패, 테이블 없음, 손상된 행 등
        log.exception("health: DB 확인 실패")
        session.rollback()
        database = "error"
    # loaded: 추론부가 production_version 을 실제로 불러와 두었는가. 여기서는 모델 파일을 읽지 않는다
    # (3~5초마다 불리는 엔드포인트이고, 불러오기는 서버 시작 때와 /predict 때 한다).
    loaded = production_version is not None and inference.loaded_version == production_version
    return HealthResponse(
        status="ok" if database == "ok" else "degraded",
        database=database,
        model=ModelStatus(production_version=production_version, loaded=loaded),
    )
