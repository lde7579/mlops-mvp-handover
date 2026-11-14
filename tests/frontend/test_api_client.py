"""frontend/api_client.py 테스트: 가짜 전송(MockTransport)으로 요청·오류 변환을 확인하고, 진짜 앱(TestClient)에도 한 번 붙여 본다."""
import json
import time

import httpx2
import pytest
from starlette.testclient import TestClient

from backend.main import create_app
from frontend.api_client import ApiClient, ApiError, ServerDown
from tests.backend.helpers import PASSWORDS, make_settings


def mock_client(handler):
    return httpx2.Client(base_url="http://api.test", transport=httpx2.MockTransport(handler))


def test_login_sends_credentials_and_returns_json():
    seen = {}

    def handler(request):
        seen["path"], seen["method"], seen["body"] = request.url.path, request.method, json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx2.Response(200, json={"access_token": "t", "token_type": "bearer", "expires_in": 60, "role": "admin", "username": "admin"})

    out = ApiClient(client=mock_client(handler)).login("admin", "pw")
    assert out["access_token"] == "t" and out["role"] == "admin"
    assert (seen["method"], seen["path"], seen["body"]) == ("POST", "/auth/login", {"username": "admin", "password": "pw"})
    assert seen["auth"] is None  # 로그인 요청에는 토큰이 없다


def test_token_is_sent_as_bearer_header():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx2.Response(200, json={"status": "ok"})

    ApiClient(token="abc", client=mock_client(handler)).health()
    assert seen["auth"] == "Bearer abc"


def test_api_error_carries_status_error_and_message():
    def handler(request):
        return httpx2.Response(401, json={"error": "unauthorized", "message": "계정 또는 비밀번호가 틀립니다"})

    with pytest.raises(ApiError) as info:
        ApiClient(client=mock_client(handler)).login("a", "b")
    assert (info.value.status, info.value.error, info.value.message) == (401, "unauthorized", "계정 또는 비밀번호가 틀립니다")


def test_non_json_error_body_still_becomes_api_error():
    with pytest.raises(ApiError) as info:
        ApiClient(client=mock_client(lambda r: httpx2.Response(502, text="Bad Gateway"))).health()
    assert info.value.status == 502 and info.value.error == "unknown" and "Bad Gateway" in info.value.message


def test_connection_failure_and_timeout_become_server_down():
    def refuse(request):
        raise httpx2.ConnectError("refused", request=request)

    def slow(request):
        raise httpx2.ReadTimeout("slow", request=request)

    with pytest.raises(ServerDown):
        ApiClient(client=mock_client(refuse)).health()
    with pytest.raises(ServerDown):
        ApiClient(client=mock_client(slow)).health()


def test_unreachable_address_fails_fast():
    """아무도 듣지 않는 주소는 제한 시간 안에 ServerDown 이 된다. 직접 만든 연결은 with 로 닫힌다."""
    start = time.perf_counter()
    with pytest.raises(ServerDown):
        with ApiClient(base_url="http://127.0.0.1:9") as client:  # 9번 포트: 보통 닫혀 있다
            client.health()
    assert time.perf_counter() - start < 3.0


def test_injected_client_is_not_closed_by_with():
    inner = mock_client(lambda r: httpx2.Response(200, json={"status": "ok"}))
    with ApiClient(client=inner) as client:
        client.health()
    assert not inner.is_closed


def test_against_the_real_app():
    """진짜 FastAPI 앱(메모리 DB)에 붙여, 프론트가 기대하는 응답 형식과 맞는지 확인한다 (프론트·백엔드 계약)."""
    app = create_app(make_settings(db_path=":memory:"))
    try:
        with TestClient(app) as http:
            anonymous = ApiClient(client=http)
            health = anonymous.health()
            assert set(health) == {"status", "database", "model"} and set(health["model"]) == {"production_version", "loaded"}
            login = anonymous.login("operator", PASSWORDS["operator"])
            assert set(login) == {"access_token", "token_type", "expires_in", "role", "username"} and login["role"] == "operator"
            with pytest.raises(ApiError) as info:
                anonymous.login("operator", "wrong")
            assert info.value.status == 401 and info.value.error == "unauthorized"
    finally:
        app.state.engine.dispose()


# ---------- 대시보드 조회 메서드 (④-1b) ----------
def recording_client(seen, body=None):
    def handler(request):
        seen.append((request.method, request.url.path, request.url.query.decode(), request.headers.get("authorization")))
        return httpx2.Response(200, json=body if body is not None else {"items": [], "total": 0})

    return ApiClient(token="tok", client=mock_client(handler))


def test_dashboard_getters_send_expected_paths_and_queries():
    seen = []
    api = recording_client(seen)
    api.simulator_status()
    api.training_runs()
    api.production_model()
    api.inspections(data_split="pool")
    api.prediction_stats()
    api.performance()
    api.approvals(status="pending")
    api.drift()
    api.deployments()
    assert [(m, p, q) for m, p, q, _ in seen] == [
        ("GET", "/simulator/status", ""), ("GET", "/training/runs", "limit=1"), ("GET", "/models/production", ""),
        ("GET", "/inspections", "data_split=pool&limit=1"), ("GET", "/predictions/stats", ""),
        ("GET", "/monitoring/performance", "bucket_seconds=3600"), ("GET", "/approvals", "status=pending&limit=1"),
        ("GET", "/monitoring/drift", ""), ("GET", "/deployments", "limit=1")]
    assert all(auth == "Bearer tok" for *_, auth in seen)


def test_since_plus_sign_is_sent_as_percent_2b():
    """api-v1.md 3-3: 시간대의 `+` 는 %2B 로 보내야 서버가 +09:00 으로 읽는다."""
    seen = []
    recording_client(seen).inspections(since="2026-11-01T00:00:00+09:00")
    query = seen[0][2]
    assert "since=2026-11-01T00%3A00%3A00%2B09%3A00" in query and "+09" not in query


def test_production_404_and_server_down_are_distinguishable():
    api = ApiClient(client=mock_client(lambda r: httpx2.Response(404, json={"error": "not_found", "message": "운영 중인 모델이 없습니다"})))
    with pytest.raises(ApiError) as info:
        api.production_model()
    assert info.value.status == 404 and info.value.error == "not_found"

    def refuse(request):
        raise httpx2.ConnectError("refused", request=request)

    with pytest.raises(ServerDown):
        ApiClient(client=mock_client(refuse)).drift()


def test_dashboard_getters_against_the_real_app():
    """진짜 앱(메모리 DB)의 응답이 view 가 읽는 키를 모두 갖고 있는지 확인한다 (프론트·백엔드 계약)."""
    from datetime import datetime

    from frontend import view

    app = create_app(make_settings(db_path=":memory:"))
    try:
        with TestClient(app) as http:
            token = ApiClient(client=http).login("operator", PASSWORDS["operator"])["access_token"]
            api = ApiClient(token=token, client=http)
            stats, perf, drift = api.prediction_stats(), api.performance(), api.drift()
            assert {"total", "ng_count", "ng_rate", "recent"} <= set(stats)
            assert {"criteria", "recent"} <= set(perf) and {"n", "accuracy", "recall"} <= set(perf["recent"]) and "recent_n" in perf["criteria"]
            assert {"model_version", "active", "n", "accuracy", "recent_n", "accuracy_threshold", "confirmed"} <= set(drift["performance_warning"])
            assert "total" in api.inspections(since=view.today_start(datetime.now().astimezone())) and "total" in api.approvals(status="pending")
            assert {"running", "drift", "last_error"} <= set(api.simulator_status())
            assert "items" in api.training_runs() and "items" in api.deployments()
            with pytest.raises(ApiError) as info:
                api.production_model()  # 아직 운영 모델이 없는 앱
            assert info.value.status == 404
    finally:
        app.state.engine.dispose()


# ---------- 검사/추론 화면 메서드 (④-2) ----------
def recording_full(seen, status=200, body=None, content=None):
    """요청을 seen 에 기록하는 가짜 서버. 연결의 기본 읽기 제한은 실제 화면과 같은 3초로 맞춘다."""
    def handler(request):
        seen.append({"method": request.method, "path": request.url.path, "query": request.url.query.decode(),
                     "body": json.loads(request.content) if request.content else None, "timeout": request.extensions["timeout"]})
        if content is not None:
            return httpx2.Response(status, content=content)
        return httpx2.Response(status, json=body if body is not None else {"running": False})

    inner = httpx2.Client(base_url="http://api.test", transport=httpx2.MockTransport(handler), timeout=httpx2.Timeout(3.0))
    return ApiClient(token="tok", client=inner)


def test_inspection_methods_send_expected_requests():
    seen = []
    api = recording_full(seen)
    api.simulator_start(0.3, 1.5)
    api.simulator_stop()
    api.simulator_generate()
    api.simulator_drift(True, "dark")
    api.simulator_drift(False)
    api.predict("INS-000001")
    api.predictions(20)
    api.inspection("INS-000001")
    assert [(s["method"], s["path"], s["query"], s["body"]) for s in seen] == [
        ("POST", "/simulator/start", "", {"defect_rate": 0.3, "interval_seconds": 1.5}),
        ("POST", "/simulator/stop", "", None),
        ("POST", "/simulator/generate", "", {"scenario": "normal"}),
        ("POST", "/simulator/drift", "", {"enabled": True, "drift_type": "dark"}),
        ("POST", "/simulator/drift", "", {"enabled": False}),  # 끌 때는 drift_type 을 보내지 않는다 (보내면 400)
        ("POST", "/predict", "", {"inspection_id": "INS-000001"}),
        ("GET", "/predictions", "limit=20", None),
        ("GET", "/inspections/INS-000001", "", None)]


def test_only_slow_calls_get_the_longer_read_timeout():
    seen = []
    api = recording_full(seen)
    api.simulator_stop()
    api.simulator_generate()
    api.predict("INS-000001")
    api.predictions(20)
    api.simulator_status()
    read = [s["timeout"]["read"] for s in seen]
    assert read[:3] == [8.0, 8.0, 8.0]  # stop·generate·predict
    assert read[3:] == [3.0, 3.0]  # 나머지는 연결의 기본 3초 그대로


def test_image_returns_png_bytes_and_404_is_api_error():
    png = b"\x89PNG\r\n\x1a\nfake"
    seen = []
    assert recording_full(seen, content=png).inspection_image("INS-000002") == png
    assert (seen[0]["method"], seen[0]["path"]) == ("GET", "/inspections/INS-000002/image")
    api = ApiClient(client=mock_client(lambda r: httpx2.Response(404, json={"error": "not_found", "message": "이미지를 찾을 수 없습니다"})))
    with pytest.raises(ApiError) as info:
        api.inspection_image("INS-000009")
    assert info.value.status == 404 and info.value.error == "not_found"


def test_slow_call_timeout_becomes_server_down():
    def slow(request):
        raise httpx2.ReadTimeout("slow", request=request)

    with pytest.raises(ServerDown):
        ApiClient(client=mock_client(slow)).simulator_stop()


# ---------- 재학습 화면 메서드 (④-3) ----------
def test_retrain_methods_send_expected_requests_with_default_timeout():
    seen = []
    api = recording_full(seen)
    api.start_training_run()
    api.training_runs(10)
    api.model("v2")
    api.inspections(data_split="pool", limit=1)
    api.approvals(limit=20)  # status 를 보내지 않는다 (처리된 요청도 받는다)
    assert [(s["method"], s["path"], s["query"], s["body"]) for s in seen] == [
        ("POST", "/training/runs", "", {"trigger": "manual"}),
        ("GET", "/training/runs", "limit=10", None),
        ("GET", "/models/v2", "", None),
        ("GET", "/inspections", "data_split=pool&limit=1", None),
        ("GET", "/approvals", "limit=20", None)]
    assert {s["timeout"]["read"] for s in seen} == {3.0}  # 계획: 읽기 제한 시간은 기본값 그대로 (점검에서 측정)


def test_retrain_409_and_server_down_are_distinguishable():
    seen = []
    api = recording_full(seen, status=409, body={"error": "conflict", "message": "재학습에 쓸 신규 데이터가 없습니다"})
    with pytest.raises(ApiError) as info:
        api.start_training_run()
    assert (info.value.status, info.value.error) == (409, "conflict")

    def refuse(request):
        raise httpx2.ReadTimeout("slow", request=request)

    with pytest.raises(ServerDown):
        ApiClient(client=mock_client(refuse)).start_training_run()


# ---------- 승인 화면 메서드 (④-4) ----------
def test_approval_methods_send_expected_requests_with_longer_timeout():
    seen = []
    api = recording_full(seen)
    api.approve_approval("APR-000002")
    api.approve_approval("APR-000002", "문제 없음")
    api.reject_approval("APR-000003", "성능 부족")
    api.approvals(status="pending", limit=20)
    api.production_model()
    assert [(s["method"], s["path"], s["query"], s["body"]) for s in seen] == [
        ("POST", "/approvals/APR-000002/approve", "", {}),  # 사유가 없으면 빈 본문 (보내지 않는다)
        ("POST", "/approvals/APR-000002/approve", "", {"reason": "문제 없음"}),
        ("POST", "/approvals/APR-000003/reject", "", {"reason": "성능 부족"}),
        ("GET", "/approvals", "status=pending&limit=20", None),
        ("GET", "/models/production", "", None)]
    assert [s["timeout"]["read"] for s in seen] == [8.0, 8.0, 8.0, 3.0, 3.0]  # 승인·거절만 8초 (첫 승인 약 5.7초)


def test_approval_errors_keep_status_and_slow_timeout_is_server_down():
    for status in (400, 403, 404, 409, 503):
        api = recording_full([], status=status, body={"error": "x", "message": f"m{status}"})
        with pytest.raises(ApiError) as info:
            api.approve_approval("APR-000002")
        assert info.value.status == status and info.value.message == f"m{status}"

    def slow(request):
        raise httpx2.ReadTimeout("slow", request=request)

    with pytest.raises(ServerDown):
        ApiClient(client=mock_client(slow)).reject_approval("APR-000002", "x")


# ---------- 모델 관리 화면 메서드 (④-5) ----------
def test_models_methods_send_expected_requests_with_rollback_timeout():
    seen = []
    api = recording_full(seen)
    api.models()
    api.models(stage="archived")
    api.deployments(limit=20)
    api.rollback_model("v1", "시연 롤백")
    assert [(s["method"], s["path"], s["query"], s["body"]) for s in seen] == [
        ("GET", "/models", "", None),  # stage 를 주지 않으면 쿼리 없음
        ("GET", "/models", "stage=archived", None),
        ("GET", "/deployments", "limit=20", None),
        ("POST", "/models/v1/rollback", "", {"reason": "시연 롤백"})]
    assert [s["timeout"]["read"] for s in seen] == [3.0, 3.0, 3.0, 8.0]  # 롤백만 8초 (새 모델을 읽은 뒤 응답)


def test_rollback_errors_keep_status_and_slow_timeout_is_server_down():
    for status in (400, 403, 404, 409, 503):
        api = recording_full([], status=status, body={"error": "x", "message": f"m{status}"})
        with pytest.raises(ApiError) as info:
            api.rollback_model("v1", "x")
        assert info.value.status == status and info.value.message == f"m{status}"

    def slow(request):
        raise httpx2.ReadTimeout("slow", request=request)

    with pytest.raises(ServerDown):
        ApiClient(client=mock_client(slow)).rollback_model("v1", "x")
