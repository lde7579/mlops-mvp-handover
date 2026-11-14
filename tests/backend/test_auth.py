"""POST /auth/login (제안 A3): 요청·응답 형식, 실패, 발급된 토큰으로 보호 경로 접근."""
import pytest

from backend.main import create_app
from backend.security import decode_token
from backend.settings import load_settings
from fastapi.testclient import TestClient
from tests.backend.helpers import PASSWORDS, SECRET, add_protected_routes, make_settings

RESPONSE_FIELDS = {"access_token", "token_type", "expires_in", "role", "username"}


def login(client, username, password):
    return client.post("/auth/login", json={"username": username, "password": password})


@pytest.mark.parametrize("username,role", [("operator", "operator"), ("admin", "admin")])
def test_login_success_for_both_accounts(client, settings, username, role):
    r = login(client, username, PASSWORDS[username])
    assert r.status_code == 200
    body = r.json()
    assert set(body) == RESPONSE_FIELDS
    assert (body["token_type"], body["role"], body["username"]) == ("bearer", role, username)
    assert body["expires_in"] == 480 * 60  # 초 단위
    assert decode_token(body["access_token"], settings.secret_key) == username


def test_issued_token_opens_protected_routes_and_role_comes_from_the_account(app, client):
    add_protected_routes(app)
    token = login(client, "operator", PASSWORDS["operator"]).json()["access_token"]
    ok = client.get("/_test/any", headers={"Authorization": f"Bearer {token}"})
    assert ok.status_code == 200 and ok.json() == {"username": "operator", "role": "operator"}
    assert client.get("/_test/admin", headers={"Authorization": f"Bearer {token}"}).status_code == 403


def test_wrong_password_and_unknown_user_get_the_same_401(client):
    wrong = login(client, "admin", "틀린-비밀번호")
    unknown = login(client, "ghost", "아무거나")
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json() == {"error": "unauthorized", "message": "계정 또는 비밀번호가 올바르지 않습니다"}
    assert wrong.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("name", ["Admin", "ADMIN", " admin", "admin ", "operator\n"])
def test_account_name_must_match_exactly(client, name):
    assert login(client, name, PASSWORDS["admin"]).status_code == 401


def test_another_accounts_password_is_rejected(client):
    assert login(client, "admin", PASSWORDS["operator"]).status_code == 401
    assert login(client, "operator", PASSWORDS["admin"]).status_code == 401


@pytest.mark.parametrize("body", [
    {}, {"username": "admin"}, {"password": "x"}, {"username": "", "password": "x"}, {"username": "admin", "password": ""},
    {"username": 123, "password": "x"}, {"username": "admin", "password": ["x"]}, {"username": None, "password": "x"},
    {"username": "a" * 65, "password": "x"}, {"username": "admin", "password": "p" * 257},
])
def test_bad_request_bodies_are_400(client, body):
    r = client.post("/auth/login", json=body)
    assert r.status_code == 400 and r.json()["error"] == "bad_request"


def test_missing_field_message_names_the_field(client):
    assert client.post("/auth/login", json={"password": "x"}).json()["message"] == "username: 필수 항목입니다"


def test_no_body_and_non_json_are_400(client):
    assert client.post("/auth/login").status_code == 400
    assert client.post("/auth/login", content=b"username=admin", headers={"Content-Type": "text/plain"}).status_code == 400


def test_get_is_405(client):
    assert client.get("/auth/login").status_code == 405


def test_response_never_contains_the_password_or_hash(client, settings):
    text = login(client, "admin", PASSWORDS["admin"]).text
    assert PASSWORDS["admin"] not in text and "pbkdf2" not in text and settings.secret_key not in text


def test_login_is_public_and_ignores_a_bad_authorization_header(client):
    r = client.post("/auth/login", json={"username": "admin", "password": PASSWORDS["admin"]},
                    headers={"Authorization": "Bearer garbage"})
    assert r.status_code == 200


def test_token_lifetime_follows_the_setting():
    app = create_app(make_settings(token_minutes=5))
    r = login(TestClient(app), "operator", PASSWORDS["operator"])
    assert r.json()["expires_in"] == 300
    app.state.engine.dispose()


def test_real_demo_accounts_from_backend_toml_can_log_in():
    """실제 backend.toml 의 시연 계정(600,000회 해시, 로그인 1회에 약 0.3초). backend/README.md 의 값과 같아야 한다."""
    real = load_settings(secret_key=SECRET)
    app = create_app(make_settings(accounts=real.accounts))
    client = TestClient(app)
    ok_operator = login(client, "operator", "operator-demo")
    ok_admin = login(client, "admin", "admin-demo")
    assert (ok_operator.status_code, ok_operator.json()["role"]) == (200, "operator")
    assert (ok_admin.status_code, ok_admin.json()["role"]) == (200, "admin")
    assert login(client, "admin", "operator-demo").status_code == 401
    app.state.engine.dispose()
