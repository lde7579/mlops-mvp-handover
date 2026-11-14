"""앱 생성 함수와 /openapi.json."""
import json

from fastapi.testclient import TestClient

import backend.main as main_module
from backend.database import session_scope
from backend.main import create_app
from backend.repository import ModelVersionRepository, Stage
from tests.backend.helpers import make_metrics, make_settings


def test_two_apps_are_independent():
    a, b = create_app(make_settings()), create_app(make_settings())
    try:
        with session_scope(a.state.session_factory) as s:
            ModelVersionRepository(s).register(make_metrics("v1"), stage=Stage.PRODUCTION)
        assert TestClient(a).get("/health").json()["model"]["production_version"] == "v1"
        assert TestClient(b).get("/health").json()["model"]["production_version"] is None
        assert a.state.settings is not b.state.settings and a.state.engine is not b.state.engine
    finally:
        a.state.engine.dispose()
        b.state.engine.dispose()


def test_app_state_holds_settings_engine_and_session_factory(app, settings):
    assert app.state.settings is settings
    assert app.state.engine is not None and app.state.session_factory is not None


def test_importing_the_module_does_not_create_an_app_or_files(tmp_path, monkeypatch):
    assert not hasattr(main_module, "app")  # create_app() 으로만 만든다 (불러오는 것만으로 DB 가 생기지 않는다)
    monkeypatch.chdir(tmp_path)
    created = create_app(make_settings())  # 메모리 DB: 어디에도 파일을 만들지 않는다
    created.state.engine.dispose()
    assert list(tmp_path.iterdir()) == []


def test_openapi_lists_both_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "get" in paths["/health"] and "post" in paths["/auth/login"]


def test_openapi_shows_400_instead_of_422(client):
    schema = client.get("/openapi.json").json()
    for path, item in schema["paths"].items():
        for method, operation in item.items():
            assert "422" not in operation["responses"], f"{method} {path}"
    login = schema["paths"]["/auth/login"]["post"]["responses"]
    assert "400" in login and "401" in login and "200" in login
    ref = login["400"]["content"]["application/json"]["schema"]["$ref"]
    assert ref.endswith("/ErrorResponse")
    components = schema["components"]["schemas"]
    assert "ErrorResponse" in components and "HTTPValidationError" not in components and "ValidationError" not in components
    assert "422" not in json.dumps(schema["paths"])


def test_openapi_error_response_has_error_and_message(client):
    props = client.get("/openapi.json").json()["components"]["schemas"]["ErrorResponse"]["properties"]
    assert set(props) == {"error", "message"}
