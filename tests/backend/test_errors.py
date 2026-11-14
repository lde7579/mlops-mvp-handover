"""오류 응답 형식: 모두 {"error": "이름", "message": "설명"} (docs/api-v1.md 1장 + 제안 A4)."""
import pytest

from tests.backend.helpers import add_error_routes, add_protected_routes


@pytest.fixture
def api(app, client):
    add_error_routes(app)
    add_protected_routes(app)
    return client


def check_shape(response, status, name):
    assert response.status_code == status
    body = response.json()
    assert set(body) == {"error", "message"} and body["error"] == name and body["message"]
    assert response.headers["content-type"].startswith("application/json")
    return body


def test_400_validation_error_is_bad_request_not_422(api):
    body = check_shape(api.get("/_test/number", params={"x": "abc"}), 400, "bad_request")
    assert body["message"].startswith("x:")
    check_shape(api.get("/_test/number"), 400, "bad_request")  # 필수 항목 없음


def test_401_unauthorized_has_www_authenticate_header(api):
    r = api.get("/_test/any")
    check_shape(r, 401, "unauthorized")
    assert r.headers["www-authenticate"] == "Bearer"


def test_403_forbidden_uses_the_given_message(api):
    body = check_shape(api.get("/_test/forbidden"), 403, "forbidden")
    assert body["message"] == "직접 쓴 메시지"


def test_404_not_found_for_unknown_path_and_method(api):
    check_shape(api.get("/no/such/path"), 404, "not_found")
    check_shape(api.post("/no/such/path"), 404, "not_found")


def test_405_method_not_allowed_has_allow_header(api):
    r = api.get("/auth/login")
    check_shape(r, 405, "method_not_allowed")
    assert "POST" in r.headers["allow"]


def test_409_conflict_uses_default_message(api):
    body = check_shape(api.get("/_test/conflict"), 409, "conflict")
    assert body["message"] == "현재 상태와 충돌합니다"


def test_500_server_error_does_not_leak_internal_details(api):
    r = api.get("/_test/boom")
    body = check_shape(r, 500, "server_error")
    text = r.text
    assert "boom-내부-비밀" not in text and "Traceback" not in text and "RuntimeError" not in text
    assert body["message"] == "서버 내부 오류가 발생했습니다"


def test_no_response_uses_the_fastapi_default_detail_format(api):
    for path in ("/_test/number", "/no/such/path", "/_test/any"):
        assert "detail" not in api.get(path).json()


def test_error_names_are_strings_not_numbers(api):
    for path in ("/_test/number", "/_test/any", "/no/such/path"):
        assert isinstance(api.get(path).json()["error"], str)


def test_validation_message_for_empty_body_and_bad_json(api):
    r = api.post("/auth/login")
    assert check_shape(r, 400, "bad_request")["message"] == "요청 본문이 필요합니다"
    r = api.post("/auth/login", content=b"{not json", headers={"Content-Type": "application/json"})
    check_shape(r, 400, "bad_request")
