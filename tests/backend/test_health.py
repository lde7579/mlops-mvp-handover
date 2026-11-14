"""GET /health: 공개, 응답 형식(제안 A2), DB 장애 시 degraded."""
from sqlalchemy.exc import OperationalError

from backend.database import session_scope
from backend.repository import ModelVersionRepository, Stage
from tests.backend.helpers import bearer, make_metrics, raw_model_version


def test_health_is_public_and_has_the_documented_shape(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "database": "ok", "model": {"production_version": None, "loaded": False}}


def test_health_gives_the_same_answer_with_or_without_a_token(client, settings):
    assert client.get("/health").json() == client.get("/health", headers=bearer(settings, "admin")).json()
    assert client.get("/health", headers={"Authorization": "Bearer garbage"}).status_code == 200  # 토큰이 틀려도 공개


def test_health_shows_the_production_version_from_the_db(app, client):
    with session_scope(app.state.session_factory) as s:
        models = ModelVersionRepository(s)
        models.register(make_metrics("v1"), stage=Stage.ARCHIVED)
        models.register(make_metrics("v2", created_minutes=1), stage=Stage.PRODUCTION)
        models.register(make_metrics("v3", created_minutes=2), stage=Stage.CANDIDATE)
    body = client.get("/health").json()
    assert body["model"] == {"production_version": "v2", "loaded": False}  # DB 에 production 이 있어도 모델 파일을 불러오기 전이라 false
    assert body["status"] == "ok" and body["database"] == "ok"


class BrokenSession:
    """DB 연결이 끊긴 세션 흉내."""

    def execute(self, *args, **kwargs):
        raise OperationalError("SELECT 1", {}, Exception("연결 실패: 비밀-경로/db.sqlite"))

    def rollback(self): pass
    def commit(self): pass
    def close(self): pass


def test_database_failure_gives_degraded_with_http_200_and_no_details(app, client):
    app.state.session_factory = BrokenSession
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "degraded", "database": "error", "model": {"production_version": None, "loaded": False}}
    assert "비밀-경로" not in r.text and "OperationalError" not in r.text


def test_corrupt_model_row_gives_degraded_not_a_crash(app, client):
    with app.state.engine.begin() as conn:  # metrics 가 비어 있는 production 행 (계약 위반)
        raw_model_version(conn, version="v1", stage="production", metrics="{}")
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "degraded" and r.json()["database"] == "error"


def test_health_response_has_no_extra_fields_and_no_secrets(client, settings):
    text = client.get("/health").text
    assert settings.secret_key not in text and "password" not in text.lower() and settings.db_path not in text


def test_health_appears_in_openapi(client):
    assert "/health" in client.get("/openapi.json").json()["paths"]
