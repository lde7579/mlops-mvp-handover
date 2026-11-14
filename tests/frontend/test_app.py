"""frontend/app.py 화면 시험: streamlit.testing 의 AppTest + 가짜 ApiClient (실제 서버·네트워크 없음)."""
import time

import pytest

import frontend.api_client as api_client
from frontend import state

HEALTH = {"status": "ok", "database": "ok", "model": {"production_version": "v1", "loaded": True}}


class FakeApi:
    """api_client.ApiClient 를 대신한다. 클래스 속성으로 응답과 실패를 정한다 (health 와 login 을 따로 실패시킬 수 있다)."""
    health_fail = None
    login_fail = None
    calls = []

    def __init__(self, base_url=None, token=None, client=None):
        self.token = token

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def health(self):
        FakeApi.calls.append("health")
        if FakeApi.health_fail:
            raise FakeApi.health_fail
        return HEALTH

    def login(self, username, password):
        FakeApi.calls.append(f"login:{username}:{password}")
        if FakeApi.login_fail:
            raise FakeApi.login_fail
        return {"access_token": "tok", "token_type": "bearer", "expires_in": 3600, "username": username,
                "role": "admin" if username == "admin" else "operator"}

    # ---- 대시보드 조회: DASH 의 기본 응답을 쓰고, fake.dash[이름] 에 예외를 넣으면 그 호출이 실패한다 ----
    def _dash(self, name):
        FakeApi.calls.append(name)
        value = FakeApi.dash[name]
        if isinstance(value, Exception):
            raise value
        return value

    def simulator_status(self):
        return self._dash("simulator_status")

    def training_runs(self, limit=1):
        return self._dash("training_runs")

    def production_model(self):
        return self._dash("production_model")

    def inspections(self, since=None, data_split=None, limit=1):
        return self._dash("pool" if data_split == "pool" else "today")

    def prediction_stats(self):
        return self._dash("prediction_stats")

    def performance(self, bucket_seconds=3600):
        return self._dash("performance")

    def approvals(self, status=None, limit=1):
        return self._dash("approvals")

    def drift(self):
        return self._dash("drift")

    def deployments(self, limit=1):
        return self._dash("deployments")


def default_dash():
    return {
        "simulator_status": {"running": True, "drift": {"enabled": False, "drift_type": None}, "last_error": None},
        "training_runs": {"items": [], "total": 0},
        "production_model": {"version": "v1"},
        "today": {"items": [], "total": 12}, "pool": {"items": [], "total": 7},
        "prediction_stats": {"total": 10, "ng_count": 2, "ng_rate": 0.2, "recent": {}},
        "performance": {"criteria": {"recent_n": 30, "accuracy_threshold": 0.95, "confirmed": True},
                        "recent": {"model_version": "v1", "n": 30, "accuracy": 0.97, "recall": 0.9, "confusion_matrix": [[1, 0], [0, 1]]}},
        "approvals": {"items": [], "total": 1},
        "drift": {"performance_warning": {"model_version": "v1", "active": False, "n": 30, "accuracy": 0.97, "recent_n": 30,
                                          "accuracy_threshold": 0.95, "confirmed": True}},
        "deployments": {"items": [], "total": 0},
    }


@pytest.fixture
def fake(monkeypatch):
    FakeApi.health_fail = FakeApi.login_fail = None
    FakeApi.dash = default_dash()
    FakeApi.calls = []
    monkeypatch.setattr(api_client, "ApiClient", FakeApi)
    return FakeApi


def entry():
    """AppTest 가 실행할 진입 함수 (app.py 를 파일로 실행하면 옆의 pages/ 폴더 때문에 두 번째 실행부터 엉뚱한 파일이 실행된다)."""
    from frontend import app

    app.main()


def new_app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_function(entry, default_timeout=15)


def texts(elements):
    return " | ".join(str(e.value) for e in elements)


def test_first_screen_asks_for_login_and_shows_status(fake):
    at = new_app().run()
    assert not at.exception
    assert texts(at.title) == "대시보드"
    assert "로그인해 주세요" in texts(at.info)
    sidebar = texts(at.sidebar.markdown)
    assert "백엔드 상태" in sidebar and "정상" in sidebar and "v1 (불러옴)" in sidebar
    assert at.sidebar.selectbox[0].options == ["운영자", "관리자"]  # 화면에는 한글 이름, 값은 operator / admin
    assert state.AUTH_KEY not in at.session_state


def test_login_success_shows_user_and_logout(fake):
    at = new_app().run()
    at.sidebar.selectbox[0].select("admin")
    at.sidebar.text_input[0].input("secret")
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert "login:admin:secret" in fake.calls
    auth = at.session_state[state.AUTH_KEY]
    assert auth["role"] == "admin" and auth["token"] == "tok" and auth["expires_at"] > time.time()
    assert "admin (관리자)" in texts(at.sidebar.markdown) and "admin (관리자)" in texts(at.caption)
    assert [b.label for b in at.sidebar.button] == ["로그아웃"]
    at.sidebar.button[0].click().run()
    assert state.AUTH_KEY not in at.session_state and "로그인해 주세요" in texts(at.info)


def test_wrong_password_shows_message_and_stays_logged_out(fake):
    fake.login_fail = api_client.ApiError(401, "unauthorized", "x")
    at = new_app().run()
    at.sidebar.text_input[0].input("bad")
    at.sidebar.button[0].click().run()
    assert "계정 또는 비밀번호가 틀렸습니다." in texts(at.sidebar.error)
    assert state.AUTH_KEY not in at.session_state


def test_server_down_is_shown_and_login_form_stays(fake):
    fake.health_fail = api_client.ServerDown("refused")
    at = new_app().run()
    assert not at.exception
    sidebar = texts(at.sidebar.markdown) + " " + texts(at.sidebar.caption)
    assert "연결 안 됨" in sidebar and "서버에 연결할 수 없습니다" in sidebar and "확인 불가" in sidebar
    assert len(at.sidebar.text_input) == 1  # 로그인 폼은 그대로


def test_expired_token_counts_as_logged_out(fake):
    at = new_app()
    at.session_state[state.AUTH_KEY] = {"token": "old", "role": "admin", "username": "admin", "expires_at": time.time() - 1}
    at.run()
    assert not at.exception
    assert state.AUTH_KEY not in at.session_state and "로그인해 주세요" in texts(at.info)


# ---------- 대시보드 본문 (④-1b) ----------
def logged_in():
    at = new_app()
    at.session_state[state.AUTH_KEY] = {"token": "tok", "role": "operator", "username": "operator", "expires_at": time.time() + 3600}
    return at


def test_dashboard_draws_stages_cards_and_green_warning(fake):
    at = logged_in().run()
    assert not at.exception
    body = texts(at.markdown)
    for stage in ("수집", "추론", "학습", "배포", "동작중"):
        assert stage in body
    for text in ("운영 모델", "v1", "12건", "20.0%", "97.0%", "승인 대기", "1건", "pool 건수", "7건"):
        assert text in body
    assert "정상: 최근 30건 정확도 97.0%" in texts(at.success)
    assert "마지막 갱신 성공" in texts(at.caption) and not at.error
    assert state.SNAPSHOT_KEY in at.session_state


def test_dashboard_hold_after_swap_and_red_warning(fake):
    w = fake.dash["drift"]["performance_warning"]
    w.update(active=None, n=4, accuracy=None, model_version="v2")
    fake.dash["deployments"] = {"items": [{"deployment_id": "DEP-1", "version": "v2", "action": "deploy", "reason": None, "actor": "admin", "at": "x"}], "total": 1}
    at = logged_in().run()
    assert "방금 모델이 v2으로 바뀌었습니다" in texts(at.markdown) and "deployments" in fake.calls and not at.success and not at.error
    w.update(active=True, n=30, accuracy=0.7)
    fake.calls.clear()
    at = logged_in().run()
    assert "성능 저하 경고" in texts(at.error) and "deployments" not in fake.calls  # 보류가 아니면 배포 기록은 읽지 않는다


def test_one_failed_call_only_breaks_its_own_card(fake):
    fake.dash["approvals"] = api_client.ApiError(500, "server_error", "승인 조회 오류")
    at = logged_in().run()
    assert not at.exception
    body = texts(at.markdown)
    assert "조회 실패" in body and "승인 조회 오류" in texts(at.caption)
    assert "12건" in body and "v1" in body  # 다른 카드는 그대로


def test_no_production_model_is_shown_as_state_not_failure(fake):
    fake.dash["production_model"] = api_client.ApiError(404, "not_found", "운영 중인 모델이 없습니다")
    at = logged_in().run()
    assert "운영 모델 없음" in texts(at.markdown) and "조회 실패" not in texts(at.markdown)


def test_server_down_keeps_last_values_with_time_and_skips_remaining_calls(fake):
    at = logged_in().run()
    assert not at.error
    fake.health_fail = api_client.ServerDown("refused")
    fake.calls.clear()
    at.run()
    assert not at.exception
    assert "서버에 연결할 수 없습니다" in texts(at.error)
    assert "기준입니다" in texts(at.caption) and "12건" in texts(at.markdown)  # 마지막 값이 시각과 함께 남는다
    assert "simulator_status" not in fake.calls  # 첫 호출(health)이 실패하면 나머지는 부르지 않는다


def test_server_down_before_any_success_shows_only_the_message(fake):
    fake.health_fail = api_client.ServerDown("refused")
    at = logged_in().run()
    assert not at.exception and "서버에 연결할 수 없습니다" in texts(at.error)
    assert "아직 성공한 갱신이 없습니다" in texts(at.caption) and "12건" not in texts(at.markdown)


def test_expired_session_401_logs_out_once(fake):
    fake.dash["approvals"] = api_client.ApiError(401, "unauthorized", "만료")
    at = logged_in().run()
    assert not at.exception
    assert state.AUTH_KEY not in at.session_state and state.SNAPSHOT_KEY not in at.session_state
    assert "로그인이 만료되었습니다" in texts(at.sidebar.error)


def test_logged_out_dashboard_makes_no_calls(fake):
    new_app().run()
    assert fake.calls == ["health"]  # 사이드바의 health 한 번뿐


# ---------- 검사/추론 페이지 등록 (④-2) ----------
def test_navigation_registers_dashboard_first_then_inspection_retrain_approvals_and_models(fake, monkeypatch):
    import streamlit as st

    seen = {}
    real = st.navigation

    def spy(pages, *args, **kwargs):
        seen["pages"] = [(p.title, p.url_path, p._default) for p in pages]
        return real(pages, *args, **kwargs)

    monkeypatch.setattr(st, "navigation", spy)
    at = logged_in().run()
    assert not at.exception
    assert seen["pages"] == [("대시보드", "", True), ("검사/추론", "inspection", False), ("재학습", "retrain", False), ("승인", "approvals", False), ("모델 관리", "models", False)]  # 대시보드가 기본 페이지 그대로
