"""3-C에서 바뀐 공통 규칙: 요청의 모르는 필드는 400 (로그인 포함), 오류 메시지 문구, backend.toml 의 collector_config."""
import json

import pytest

from backend.settings import DEFAULT_COLLECTOR_CONFIG, PROJECT_ROOT, SettingsError, load_settings
from tests.backend.helpers import PASSWORDS, SECRET, bearer

GOOD_HASH = "pbkdf2_sha256$1000$c2FsdHNhbHRzYWx0c2FsdA$ZGlnZXN0ZGlnZXN0ZGlnZXN0ZGlnZXN0ZGlnZXN0ZGlnZXN0"


# ---------- 로그인도 모르는 필드는 400 ----------

def test_login_with_unknown_field_is_400_and_names_the_field(client):
    r = client.post("/auth/login", json={"username": "admin", "password": PASSWORDS["admin"], "role": "admin"})
    assert r.status_code == 400 and r.json()["error"] == "bad_request" and "role" in r.json()["message"]
    assert "access_token" not in r.text


def test_login_still_works_without_extra_fields(client):
    assert client.post("/auth/login", json={"username": "admin", "password": PASSWORDS["admin"]}).status_code == 200


# ---------- 오류 메시지는 한국어 문구 ----------

@pytest.mark.parametrize("path, params, expected", [
    ("/inspections", {"limit": 0}, "limit: 허용 범위보다 작습니다"),
    ("/inspections", {"limit": 1001}, "limit: 허용 범위보다 큽니다"),
    ("/inspections", {"limit": "abc"}, "limit: 정수여야 합니다"),
    ("/inspections", {"scenario": "bogus"}, "scenario: 허용된 값이 아닙니다"),
    ("/inspections", {"data_split": "bogus"}, "data_split: 허용된 값이 아닙니다"),
])
def test_query_error_messages(client, settings, path, params, expected):
    r = client.get(path, params=params, headers=bearer(settings, "operator"))
    assert r.status_code == 400 and r.json()["message"] == expected


def test_body_error_messages(client, settings):
    headers = bearer(settings, "operator")
    assert client.post("/simulator/generate", json={"scenario": "x"}, headers=headers).json()["message"] == "scenario: 허용된 값이 아닙니다"
    assert client.post("/simulator/generate", json={"extra": 1}, headers=headers).json()["message"] == "extra: 허용되지 않는 항목입니다"


def test_openapi_lists_the_new_endpoints_with_400_not_422(client):
    spec = client.get("/openapi.json").json()
    for path, method in (("/simulator/generate", "post"), ("/inspections", "get"), ("/inspections/{inspection_id}", "get"),
                         ("/inspections/{inspection_id}/image", "get")):
        responses = spec["paths"][path][method]["responses"]
        assert "422" not in responses and "400" in responses
    assert "201" in spec["paths"]["/simulator/generate"]["post"]["responses"]
    assert "image/png" in spec["paths"]["/inspections/{inspection_id}/image"]["get"]["responses"]["200"]["content"]


# ---------- backend.toml: collector_config ----------

def write_toml(tmp_path, extra=""):
    text = ('db_path = "data/test.db"\ntoken_minutes = 480\n' + extra +
            f'[[accounts]]\nusername = "operator"\nrole = "operator"\npassword_hash = {json.dumps(GOOD_HASH)}\n')
    path = tmp_path / "backend.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_real_backend_toml_points_to_the_real_collector_config():
    s = load_settings(secret_key=SECRET)
    assert s.collector_config == str(DEFAULT_COLLECTOR_CONFIG.resolve()) and s.root == str(PROJECT_ROOT.resolve())
    assert DEFAULT_COLLECTOR_CONFIG.is_file()


def test_collector_config_is_resolved_against_root(tmp_path):
    s = load_settings(write_toml(tmp_path, 'collector_config = "conf/c.toml"\n'), secret_key=SECRET, root=tmp_path)
    assert s.collector_config == str((tmp_path / "conf" / "c.toml").resolve()) and s.root == str(tmp_path.resolve())


def test_collector_config_may_be_omitted(tmp_path):
    s = load_settings(write_toml(tmp_path), secret_key=SECRET, root=tmp_path)
    assert s.collector_config == str((tmp_path / "parts" / "collector" / "collector.toml").resolve())


@pytest.mark.parametrize("value", ['"/abs/c.toml"', '"C:\\\\abs\\\\c.toml"', '""', "5", "true"])
def test_bad_collector_config_is_rejected(tmp_path, value):
    with pytest.raises(SettingsError, match="collector_config"):
        load_settings(write_toml(tmp_path, f"collector_config = {value}\n"), secret_key=SECRET, root=tmp_path)
