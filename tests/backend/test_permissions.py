"""권한 확인 의존성(CurrentUser, require_role): 3-C 이후 엔드포인트가 재사용하는 것."""
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.security import Role
from tests.backend.helpers import OTHER_SECRET, SECRET, add_protected_routes, bearer, make_settings

ROUTES = ["/_test/any", "/_test/operator", "/_test/admin"]


@pytest.fixture
def api(app, client):
    add_protected_routes(app)
    return client


def expect_401(response):
    assert response.status_code == 401
    assert response.json()["error"] == "unauthorized"
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("route", ROUTES)
def test_no_authorization_header_is_401(api, route):
    expect_401(api.get(route))


@pytest.mark.parametrize("header", ["Basic dXNlcjpwdw==", "Bearer", "Bearer ", "Token abc", "abc", "Bearer garbage",
                                    "Bearer a.b.c"])
def test_malformed_authorization_header_is_401(api, header):
    expect_401(api.get("/_test/any", headers={"Authorization": header}))


def test_forged_tampered_and_expired_tokens_are_401(api, settings):
    expect_401(api.get("/_test/any", headers=bearer(settings, "admin", secret=OTHER_SECRET)))  # 다른 키로 서명
    head, payload, sig = bearer(settings, "operator")["Authorization"].split(" ")[1].split(".")
    forged = jwt.encode({"sub": "admin", "iat": 1, "exp": 9999999999}, OTHER_SECRET, algorithm="HS256").split(".")[1]
    expect_401(api.get("/_test/admin", headers={"Authorization": f"Bearer {head}.{forged}.{sig}"}))  # 내용만 바꿈
    old = datetime.now(timezone.utc) - timedelta(hours=9)
    expect_401(api.get("/_test/any", headers=bearer(settings, "admin", now=old)))  # 만료


def test_alg_none_token_is_401(api):
    unsigned = jwt.encode({"sub": "admin", "iat": 1, "exp": 9999999999}, key=None, algorithm="none")
    expect_401(api.get("/_test/admin", headers={"Authorization": f"Bearer {unsigned}"}))


def test_scheme_name_is_case_insensitive(api, settings):
    token = bearer(settings, "operator")["Authorization"].split(" ")[1]
    assert api.get("/_test/any", headers={"Authorization": f"bearer {token}"}).status_code == 200


def test_operator_passes_operator_routes_but_not_admin_only(api, settings):
    h = bearer(settings, "operator")
    assert api.get("/_test/any", headers=h).json() == {"username": "operator", "role": "operator"}
    assert api.get("/_test/operator", headers=h).status_code == 200
    r = api.get("/_test/admin", headers=h)
    assert r.status_code == 403 and r.json() == {"error": "forbidden", "message": "이 작업은 관리자만 할 수 있습니다"}
    assert "www-authenticate" not in r.headers  # 403 은 인증 실패가 아니다


@pytest.mark.parametrize("route", ROUTES)
def test_admin_passes_every_route(api, settings, route):
    r = api.get(route, headers=bearer(settings, "admin"))
    assert r.status_code == 200 and r.json() == {"username": "admin", "role": "admin"}


def test_role_hierarchy_is_admin_over_operator():
    assert [r.value for r in sorted(Role, key=lambda r: r.value)] == ["admin", "operator"]


def test_token_of_an_account_removed_from_settings_is_401():
    full = make_settings()
    only_operator = make_settings(accounts=tuple(a for a in full.accounts if a.username == "operator"))
    app = create_app(only_operator)
    add_protected_routes(app)
    client = TestClient(app, raise_server_exceptions=False)
    expect_401(client.get("/_test/any", headers=bearer(full, "admin")))  # 서명은 맞지만 계정이 없다
    assert client.get("/_test/any", headers=bearer(full, "operator")).status_code == 200
    app.state.engine.dispose()


def test_role_is_taken_from_settings_not_from_the_token():
    # 토큰에 role 을 넣어도 무시한다 (설정이 바뀌면 바로 반영)
    settings = make_settings()
    app = create_app(settings)
    add_protected_routes(app)
    client = TestClient(app, raise_server_exceptions=False)
    claims = {"sub": "operator", "role": "admin", "iat": int(datetime.now(timezone.utc).timestamp()),
              "exp": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp())}
    h = {"Authorization": "Bearer " + jwt.encode(claims, SECRET, algorithm="HS256")}
    assert client.get("/_test/admin", headers=h).status_code == 403
    app.state.engine.dispose()


def test_protected_routes_show_up_with_a_bearer_security_scheme_in_openapi(app, api):
    schema = api.get("/openapi.json").json()
    assert schema["components"]["securitySchemes"]
    assert any(s.get("scheme") == "bearer" for s in schema["components"]["securitySchemes"].values())
