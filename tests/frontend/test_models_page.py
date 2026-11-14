"""frontend/pages/models.py 화면 시험: streamlit.testing 의 AppTest + 가짜 ApiClient (실제 서버·네트워크·data/·models/ 없음)."""
import copy
import time

import pytest

import frontend.api_client as api_client
from frontend import state, view
from frontend.pages import models as page
from tests.frontend.test_view_models import CREATED, model

AT = "2026-11-01T09:00:00.000000Z"


class FakeApi:
    """api_client.ApiClient 를 대신한다. world 가 '서버의 실제 상태'이고, fail[이름] 에 예외를 넣으면 그 호출이 실패한다."""
    world = {}
    calls = []
    args = {}
    fail = {}

    def __init__(self, base_url=None, token=None, client=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def _do(self, name, **args):
        FakeApi.calls.append(name)
        FakeApi.args.setdefault(name, []).append(args)
        if name in FakeApi.fail:
            raise FakeApi.fail[name]

    def health(self):
        self._do("health")
        prod = next((m["version"] for m in FakeApi.world["models"] if m["stage"] == "production"), None)
        return {"status": "ok", "database": "ok", "model": {"production_version": prod, "loaded": True}}

    def models(self, stage=None):
        self._do("models", stage=stage)
        return copy.deepcopy({"items": FakeApi.world["models"], "total": len(FakeApi.world["models"])})  # 서버는 매번 새 JSON 을 준다

    def deployments(self, limit=1):
        self._do("deployments", limit=limit)
        items = FakeApi.world["deployments"]
        return copy.deepcopy({"items": items[:limit], "total": len(items)})

    def rollback_model(self, version, reason):
        self._do("rollback", version=version, reason=reason)
        if not reason:
            raise api_client.ApiError(400, "bad_request", "reason 은 필수입니다")
        target = next((m for m in FakeApi.world["models"] if m["version"] == version), None)
        if target is None:
            raise api_client.ApiError(404, "not_found", "모델 버전을 찾을 수 없습니다")
        if target["stage"] != "archived":
            raise api_client.ApiError(409, "conflict", "archived 가 아닙니다")
        for m in FakeApi.world["models"]:
            if m["stage"] == "production":
                m["stage"] = "archived"
        target.update(stage="production", approved_by="admin", approved_at=AT)
        FakeApi.world["deployments"].insert(0, {"deployment_id": f"DEP-{len(FakeApi.world['deployments']) + 1:06d}", "version": version,
                                                "action": "rollback", "reason": reason, "actor": "admin", "at": AT})
        return {"new_version": "v9", "old_version": "v9", "deployment": {"version": "v9"}}  # 일부러 틀린 본문: 화면은 응답이 아니라 서버를 다시 읽은 값을 믿어야 한다


@pytest.fixture
def fake(monkeypatch):
    FakeApi.world = {"models": [model("v3", "production", 1.0, "admin", CREATED), model("v2", "rejected", 0.914), model("v1", "archived", 0.854, "admin", CREATED)],
                     "deployments": [{"deployment_id": "DEP-000001", "version": "v3", "action": "deploy", "reason": None, "actor": "admin", "at": CREATED}]}
    FakeApi.calls, FakeApi.args, FakeApi.fail = [], {}, {}
    monkeypatch.setattr(api_client, "ApiClient", FakeApi)
    return FakeApi


def entry():
    from frontend.pages import models as p

    p.render()


def new_page(role="admin", logged_in=True):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(entry, default_timeout=15)
    if logged_in:
        at.session_state[state.AUTH_KEY] = {"token": "tok", "role": role, "username": role, "expires_at": time.time() + 3600}
    return at


def texts(elements):
    return " | ".join(str(e.value) for e in elements)


def click(at, reason=None):
    if reason is not None:
        at.text_input(key=page.REASON_KEY).set_value(reason)
    at.button(key=page.ROLLBACK_BUTTON_KEY).click().run()
    assert not at.exception


def opened(role="admin"):
    at = new_page(role).run()
    assert not at.exception
    FakeApi.calls.clear()
    FakeApi.args.clear()
    return at


def test_logged_out_makes_no_calls(fake):
    at = new_page(logged_in=False).run()
    assert not at.exception and "로그인해 주세요" in texts(at.info) and fake.calls == [] and not at.button


def test_first_draw_reads_models_and_deployments_only_and_shows_server_values(fake):
    at = new_page().run()
    assert not at.exception and fake.calls == ["models", "deployments"] and fake.args["deployments"] == [{"limit": 20}]
    body = texts(at.markdown)
    assert "운영 모델 v3" in body and ":green[운영 중]" in body and "거절됨" in body and "보관(archived)" in body
    assert "v3 (A)" in body and "v2 (B)" in body and "혼동행렬" in body and "TN 400" in body  # 기본 비교: 운영 모델 vs 직전 버전
    assert "DEP-000001" in body and "배포" in body and "전체 배포·롤백 1건 중 최근 1건" in texts(at.caption)
    assert at.selectbox(key=page.TARGET_KEY).options == ["v1"]  # 롤백 선택지는 archived 만 (rejected v2, 운영 중 v3 없음)
    fake.calls.clear()
    at.run()
    assert fake.calls == ["models", "deployments"]  # 갱신당 2번


def test_operator_sees_everything_but_rollback_is_disabled_with_hint(fake):
    at = new_page("operator").run()
    assert not at.exception
    assert at.button(key=page.ROLLBACK_BUTTON_KEY).disabled and at.text_input(key=page.REASON_KEY).disabled and at.selectbox(key=page.TARGET_KEY).disabled
    assert view.ROLLBACK_PERMISSION_TEXT in texts(at.caption) and "운영 모델 v3" in texts(at.markdown)


def test_admin_rollback_posts_once_rereads_server_and_ignores_response_body(fake):
    at = opened()
    click(at, "시연 롤백")
    assert fake.calls == ["rollback", "models", "deployments", "models", "deployments"]  # 뒤쪽 두 번은 사이드바 갱신용 전체 다시 실행의 첫 읽기
    assert fake.args["rollback"] == [{"version": "v1", "reason": "시연 롤백"}] and len(fake.calls) <= 5
    assert "롤백 완료: v1이 운영 모델입니다" in texts(at.success) and "v9" not in texts(at.success)  # 틀린 응답 본문(v9)은 무시
    body = texts(at.markdown)
    assert "운영 모델 v1" in body and "시연 롤백" in body and "롤백" in body
    assert at.session_state[state.MODELS_SNAPSHOT_KEY]["snap"]["models"]["data"]["items"][2]["stage"] == "production"
    assert state.MODELS_INFLIGHT_KEY not in at.session_state  # 서버가 처리됐다고 확인하면 표식을 지운다
    assert at.selectbox(key=page.TARGET_KEY).options == ["v3"]  # 이제 v3 가 archived
    assert at.text_input(key=page.REASON_KEY).value == ""  # 사유 입력칸은 비워진다


def test_empty_reason_goes_to_server_and_400_is_shown_with_server_message(fake):
    at = opened()
    click(at, "   ")
    assert fake.args["rollback"] == [{"version": "v1", "reason": ""}]  # 화면이 미리 막지 않고 서버 규칙(400)을 따른다
    assert "롤백 사유를 입력해 주세요" in texts(at.error) and "서버 메시지: reason 은 필수입니다" in texts(at.error)
    assert fake.world["models"][0]["stage"] == "production" and not at.button(key=page.ROLLBACK_BUTTON_KEY).disabled


def test_server_403_404_and_401(fake):
    at = opened()
    fake.fail["rollback"] = api_client.ApiError(403, "forbidden", "x")
    click(at, "사유")
    assert "이 계정에는 롤백 권한이 없습니다." in texts(at.error)
    fake.fail["rollback"] = api_client.ApiError(404, "not_found", "x")
    click(at, "사유")
    assert "롤백할 버전을 찾을 수 없습니다" in texts(at.warning)
    fake.fail["rollback"] = api_client.ApiError(401, "unauthorized", "만료")
    click(at, "사유")
    assert state.AUTH_KEY not in at.session_state and state.MODELS_SNAPSHOT_KEY not in at.session_state
    assert at.session_state[state.LOGIN_ERROR_KEY] == page.SESSION_EXPIRED


def test_409_already_production_and_other_stage_are_told_apart_by_reread_state(fake):
    at = opened()
    fake.fail["rollback"] = api_client.ApiError(409, "conflict", "아무 글자")
    fake.world["models"][0]["stage"], fake.world["models"][2]["stage"] = "archived", "production"  # 그 사이 다른 관리자가 v1 로 롤백
    click(at, "사유")
    assert "v1은 이미 운영 중인 버전입니다" in texts(at.info)
    at.run()
    assert at.selectbox(key=page.TARGET_KEY).value == "v3"  # 선택지가 다시 읽은 archived 로 바뀐다 (입력칸은 직전에 읽은 값으로 그려서 한 번 늦게 바뀐다)
    fake.world["models"][0]["stage"] = "rejected"  # 그 사이 v3 가 거절됨 상태로 읽힘
    click(at, "사유")
    assert "v3은 지금 거절됨 상태라서 롤백할 수 없습니다" in texts(at.warning)


def test_503_keeps_current_production_and_says_what_server_shows(fake):
    at = opened()
    fake.fail["rollback"] = api_client.ApiError(503, "service_unavailable", "모델 v1 을 불러오지 못해 롤백하지 않았습니다")
    click(at, "사유")
    text = texts(at.error)
    assert "롤백하지 않았습니다" in text and "기존 운영 모델을 계속 사용합니다" in text and "운영 모델은 v3" in text
    assert "운영 모델 v3" in texts(at.markdown) and not at.button(key=page.ROLLBACK_BUTTON_KEY).disabled


def test_server_down_on_post_keeps_marker_and_blocks_duplicate_for_a_while(fake):
    at = opened()
    fake.fail["rollback"] = api_client.ServerDown("timeout")
    click(at, "사유")
    assert view.SERVER_DOWN_ROLLBACK_TEXT in texts(at.warning)
    assert at.session_state[state.MODELS_INFLIGHT_KEY]["version"] == "v1"
    fake.args.clear()
    fake.fail.clear()
    at.run()
    assert at.button(key=page.ROLLBACK_BUTTON_KEY).disabled and "버튼을 잠갔습니다" in texts(at.caption)
    assert "rollback" not in fake.args  # 15초 안에는 같은 요청을 다시 보내지 않는다
    at.session_state[state.MODELS_INFLIGHT_KEY]["ts"] = time.time() - view.INFLIGHT_SECONDS - 1  # 15초가 지나면 무시
    at.run()
    assert not at.button(key=page.ROLLBACK_BUTTON_KEY).disabled


def test_send_is_blocked_by_marker_even_if_button_event_arrives(fake):
    at = opened()
    at.session_state[state.MODELS_INFLIGHT_KEY] = {"version": "v1", "ts": time.time()}
    at.run()
    assert at.button(key=page.ROLLBACK_BUTTON_KEY).disabled and "rollback" not in fake.args


def test_read_server_down_keeps_last_values_grey(fake):
    at = opened()
    fake.fail["models"] = api_client.ServerDown("refused")
    at.run()
    assert not at.exception and "서버에 연결할 수 없습니다" in texts(at.error) and "기준입니다" in texts(at.caption)
    assert ":gray[운영 모델 v3]" in texts(at.markdown)
    fake.fail.clear()
    at.run()
    assert not at.error and ":green[운영 모델 v3]" in texts(at.markdown)


def test_no_archived_version_shows_notice_and_no_button(fake):
    fake.world["models"] = [model("v2", "production", 1.0, "admin", CREATED), model("v1", "rejected")]
    at = new_page().run()
    assert not at.exception and view.NO_ARCHIVED_TEXT in texts(at.caption) and not at.button


def test_compare_choice_changes_table_and_falls_back_when_version_disappears(fake):
    at = new_page().run()
    at.selectbox(key=page.COMPARE_B_KEY).select("v1").run()
    assert not at.exception and "v1 (B)" in texts(at.markdown) and "v2 (B)" not in texts(at.markdown)
    del fake.world["models"][2]  # 고른 v1 이 목록에서 사라지면 기본값으로 되돌린다
    at.run()
    assert not at.exception and at.selectbox(key=page.COMPARE_B_KEY).value == "v2"


def test_single_version_shows_only_its_metrics(fake):
    fake.world["models"] = [model("v1", "production", 1.0, "admin", CREATED)]
    at = new_page().run()
    assert not at.exception and "비교할 대상이 없습니다" in texts(at.caption)
    assert "| 값 |" in texts(at.markdown) and "(B)" not in texts(at.markdown)


def test_model_list_error_is_shown_without_crash(fake):
    fake.fail["models"] = api_client.ApiError(500, "server_error", "목록 오류")
    at = new_page().run()
    assert not at.exception and "모델 목록을 읽지 못했습니다" in texts(at.markdown) and "목록 오류" in texts(at.markdown)


def test_logout_clears_models_state(fake):
    at = opened()
    click(at, "x")
    assert state.MODELS_SNAPSHOT_KEY in at.session_state and state.MODELS_NOTICE_KEY in at.session_state
    from streamlit.testing.v1 import AppTest

    def out():
        from frontend import state as s

        s.logout()

    check = AppTest.from_function(out)
    for key in state.MODELS_KEYS:
        check.session_state[key] = "x"
    check.run()
    assert not any(key in check.session_state for key in state.MODELS_KEYS)


def entry_with_sidebar():
    from frontend import app
    from frontend.pages import models as p

    app.sidebar()  # 실제 app.py 와 같은 사이드바 (운영 모델 줄은 GET /health 를 다시 읽을 때만 바뀐다)
    p.render()


def test_rollback_reruns_whole_script_once_so_sidebar_shows_new_model(fake):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(entry_with_sidebar, default_timeout=15)
    at.session_state[state.AUTH_KEY] = {"token": "tok", "role": "admin", "username": "admin", "expires_at": time.time() + 3600}
    at.run()
    assert not at.exception and "v3 (불러옴)" in texts(at.sidebar.markdown)
    fake.calls.clear()
    fake.args.clear()
    at.text_input(key=page.REASON_KEY).set_value("시연 롤백")
    at.button(key=page.ROLLBACK_BUTTON_KEY).click().run()
    assert not at.exception
    assert fake.calls.count("health") == 2 and len(fake.args["rollback"]) == 1  # 클릭 실행 1번 + st.rerun() 1번 (되풀이되지 않음)
    assert "v1 (불러옴)" in texts(at.sidebar.markdown) and "v3 (불러옴)" not in texts(at.sidebar.markdown)
    assert "롤백 완료" in texts(at.success)  # 다시 실행한 뒤에도 알림이 남는다
    fake.calls.clear()
    at.run()
    assert fake.calls.count("health") == 1 and "rollback" not in fake.calls  # 그 다음 갱신은 평범한 1번
