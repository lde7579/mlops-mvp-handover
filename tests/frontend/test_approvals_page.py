"""frontend/pages/approvals.py 화면 시험: streamlit.testing 의 AppTest + 가짜 ApiClient (실제 서버·네트워크·data/·models/ 없음)."""
import copy
import time

import pytest

import frontend.api_client as api_client
from frontend import state, view
from frontend.pages import approvals as page
from tests.frontend.test_view_approvals import item
from tests.frontend.test_view_retrain import gate, metrics, run

HANDLED = "2026-11-01T08:05:00.000000Z"


def model(version, acc_drift):
    return {"version": version, "stage": "STALE-STAGE", "metrics": metrics(1.0, acc_drift)}


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
        return {"status": "ok", "database": "ok", "model": {"production_version": FakeApi.world["production"], "loaded": True}}

    def approvals(self, status=None, limit=1):
        self._do("approvals", status=status, limit=limit)
        items = FakeApi.world["approvals"]
        return copy.deepcopy({"items": items[:limit], "total": len(items)})  # 서버는 매번 새 JSON 을 준다 (화면이 쥔 값이 world 와 같은 객체면 안 된다)

    def training_runs(self, limit=1):
        self._do("runs", limit=limit)
        return copy.deepcopy({"items": FakeApi.world["runs"][:limit], "total": len(FakeApi.world["runs"])})

    def production_model(self):
        self._do("production")
        if FakeApi.world["production"] is None:
            raise api_client.ApiError(404, "not_found", "운영 중인 모델이 없습니다")
        return {"version": FakeApi.world["production"], "stage": "production"}

    def model(self, version):
        self._do("model", version=version)
        return copy.deepcopy(FakeApi.world["models"][version])

    def _target(self):
        return next(i for i in FakeApi.world["approvals"] if i["status"] == "pending")

    def approve_approval(self, approval_id, reason=None):
        self._do("approve", approval_id=approval_id, reason=reason)
        found = next((i for i in FakeApi.world["approvals"] if i["approval_id"] == approval_id), None)
        if found is None:
            raise api_client.ApiError(404, "not_found", "없는 요청")
        if found["status"] != "pending" or FakeApi.world["baseline"] != FakeApi.world["production"]:
            raise api_client.ApiError(409, "conflict", "처리할 수 없는 요청")
        found.update(status="approved", handled_by="admin", handled_at=HANDLED, reason=reason)
        FakeApi.world["production"] = found["model_version"]
        return {"approval": {"status": "pending"}, "new_version": "v9"}  # 일부러 틀린 본문: 화면은 응답이 아니라 서버를 다시 읽은 값을 믿어야 한다

    def reject_approval(self, approval_id, reason):
        self._do("reject", approval_id=approval_id, reason=reason)
        if not reason:
            raise api_client.ApiError(400, "bad_request", "reason 은 필수입니다")
        found = next(i for i in FakeApi.world["approvals"] if i["approval_id"] == approval_id)
        if found["status"] != "pending":
            raise api_client.ApiError(409, "conflict", "이미 처리됨")
        found.update(status="rejected", handled_by="admin", handled_at=HANDLED, reason=reason)
        return {"stage": "rejected"}


@pytest.fixture
def fake(monkeypatch):
    FakeApi.world = {"approvals": [item(2)], "production": "v1", "baseline": "v1",
                     "runs": [run(result="v2", n=2, gate_result=gate(baseline="v1"))],
                     "models": {"v1": model("v1", 0.854), "v2": model("v2", 0.97)}}
    FakeApi.calls, FakeApi.args, FakeApi.fail = [], {}, {}
    monkeypatch.setattr(api_client, "ApiClient", FakeApi)
    return FakeApi


def entry():
    from frontend.pages import approvals as p

    p.render()


def new_page(role="admin", logged_in=True):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(entry, default_timeout=15)
    if logged_in:
        at.session_state[state.AUTH_KEY] = {"token": "tok", "role": role, "username": role, "expires_at": time.time() + 3600}
    return at


def texts(elements):
    return " | ".join(str(e.value) for e in elements)


def click(at, key, reason=None):
    if reason is not None:
        at.text_input(key=page.REASON_KEY).set_value(reason)
    at.button(key=key).click().run()
    assert not at.exception


def opened(role="admin"):
    at = new_page(role).run()
    assert not at.exception
    FakeApi.calls.clear()
    FakeApi.args.clear()
    return at


def test_logged_out_makes_no_calls_and_no_pending_reads_only_the_list(fake):
    at = new_page(logged_in=False).run()
    assert not at.exception and "로그인해 주세요" in texts(at.info) and fake.calls == []
    fake.world["approvals"] = [item(1, "v1", "approved", "admin", handled_at=HANDLED)]
    at = new_page().run()
    assert not at.exception and fake.calls == ["approvals"]  # 대기가 없으면 학습 이력·운영 모델은 읽지 않는다
    assert fake.args["approvals"] == [{"status": None, "limit": 20}]
    assert "승인 대기 중인 요청이 없습니다" in texts(at.caption) and ":green[0건]" in texts(at.markdown)
    assert "승인됨" in texts(at.markdown) and "admin" in texts(at.markdown) and not at.button


def test_operator_sees_everything_but_buttons_are_disabled_with_hint(fake):
    at = new_page("operator").run()
    assert not at.exception and fake.calls == ["approvals", "runs", "production", "model", "model"]  # 처음: 5번
    assert fake.args["runs"] == [{"limit": 10}] and [a["version"] for a in fake.args["model"]] == ["v2", "v1"]
    assert at.button(key=page.APPROVE_BUTTON_KEY).disabled and at.button(key=page.REJECT_BUTTON_KEY).disabled
    assert at.text_input(key=page.REASON_KEY).disabled and view.PERMISSION_TEXT in texts(at.caption)
    body = texts(at.markdown)
    assert "APR-000002" in body and "G1" in body and "정확도" in body and "혼동행렬" in body and "게이트가 비교한 기준 모델: v1" in body
    assert "현재 운영 모델: v1" in texts(at.caption) and ":orange[1건]" in body
    fake.calls.clear()
    at.run()
    assert fake.calls == ["approvals", "runs", "production"]  # 모델 상세는 한 번만 받는다 (캐시)


def test_admin_buttons_enabled_and_cached_stage_is_never_shown(fake):
    at = new_page().run()
    assert not at.button(key=page.APPROVE_BUTTON_KEY).disabled and not at.button(key=page.REJECT_BUTTON_KEY).disabled
    assert view.PERMISSION_TEXT not in texts(at.caption)
    everything = texts(at.markdown) + texts(at.caption) + texts(at.success) + texts(at.info) + texts(at.warning)
    assert "STALE-STAGE" not in everything  # 캐시에 든 stage 는 낡을 수 있어 쓰지 않는다


def test_approve_posts_once_rereads_server_and_ignores_response_body(fake):
    at = opened()
    click(at, page.APPROVE_BUTTON_KEY, "지표 개선 확인")
    assert fake.calls == ["approve", "approvals", "production", "approvals"]  # 마지막 approvals 는 사이드바 갱신용 전체 다시 실행의 첫 읽기. 처리된 뒤에는 대기가 없어 학습 이력은 읽지 않고, 결과 확인용으로 운영 모델은 읽는다
    assert fake.args["approve"] == [{"approval_id": "APR-000002", "reason": "지표 개선 확인"}]
    assert len(fake.calls) <= 6 and len(fake.args["approve"]) == 1  # 사이드바용 st.rerun() 뒤에도 같은 요청을 다시 보내지 않는다 (POST 정확히 1번)
    assert "승인 완료: v2이 운영 모델입니다" in texts(at.success) and "v9" not in texts(at.success)  # 틀린 응답 본문(v9)은 무시
    body = texts(at.markdown)
    assert "승인됨" in body and "지표 개선 확인" in body and ":green[0건]" in body
    assert at.session_state[state.APPROVAL_SNAPSHOT_KEY]["snap"]["approvals"]["data"]["items"][0]["status"] == "approved"
    assert state.APPROVAL_INFLIGHT_KEY not in at.session_state  # 서버가 처리됐다고 확인하면 표식을 지운다


def test_approve_without_reason_sends_no_reason(fake):
    at = opened()
    click(at, page.APPROVE_BUTTON_KEY)
    assert fake.args["approve"] == [{"approval_id": "APR-000002", "reason": None}]


def test_reject_needs_reason_server_400_then_reject_with_reason(fake):
    at = opened()
    click(at, page.REJECT_BUTTON_KEY, "   ")
    assert fake.args["reject"] == [{"approval_id": "APR-000002", "reason": ""}]  # 화면이 미리 막지 않고 서버 규칙(400)을 따른다
    assert "거절 사유를 입력해 주세요" in texts(at.error) and "서버 메시지: reason 은 필수입니다" in texts(at.error)
    assert fake.world["approvals"][0]["status"] == "pending"  # 400 이면 상태는 그대로
    assert not at.button(key=page.REJECT_BUTTON_KEY).disabled  # 확정된 오류는 바로 다시 시도할 수 있다
    click(at, page.REJECT_BUTTON_KEY, "G3 근처라 불안함 | [메모]")
    assert "거절 완료: 후보 v2" in texts(at.success)
    assert "G3 근처라 불안함 \\| \\[메모\\]" in texts(at.markdown)  # 이력의 사유는 서식 문자를 무해화해서 보여 준다
    assert fake.world["production"] == "v1"


def test_server_403_404_and_401(fake):
    at = opened()
    fake.fail["approve"] = api_client.ApiError(403, "forbidden", "x")
    click(at, page.APPROVE_BUTTON_KEY)
    assert "이 계정에는 승인 권한이 없습니다." in texts(at.error)
    fake.fail["approve"] = api_client.ApiError(404, "not_found", "x")
    click(at, page.APPROVE_BUTTON_KEY)
    assert "승인 요청을 찾을 수 없습니다" in texts(at.warning)
    fake.fail["approve"] = api_client.ApiError(401, "unauthorized", "만료")
    click(at, page.APPROVE_BUTTON_KEY)
    assert state.AUTH_KEY not in at.session_state and state.APPROVAL_SNAPSHOT_KEY not in at.session_state
    assert at.session_state[state.LOGIN_ERROR_KEY] == page.SESSION_EXPIRED


def test_409_already_handled_and_baseline_mismatch_are_told_apart(fake):
    at = opened()
    fake.world["approvals"][0].update(status="rejected", handled_by="admin", handled_at=HANDLED, reason="다른 관리자")  # 그 사이 다른 사람이 처리
    click(at, page.APPROVE_BUTTON_KEY)
    assert "이미 처리된 요청입니다" in texts(at.info) and "거절됨" in texts(at.info)
    fake.world["approvals"] = [item(3, "v3")]
    fake.world["runs"] = [run(result="v3", n=3, gate_result=gate(baseline="v1"))]
    fake.world["models"]["v3"] = model("v3", 0.96)
    fake.world["production"] = "v2"  # 다른 후보가 먼저 승인되어 운영 모델이 바뀜
    at = new_page().run()
    assert "승인하면 서버가 거절(409)합니다" in texts(at.warning)  # 누르기 전에도 미리 경고
    assert not at.button(key=page.APPROVE_BUTTON_KEY).disabled  # 막지 않고 서버가 판단한다
    click(at, page.APPROVE_BUTTON_KEY)
    assert "기준 모델(v1)과 현재 운영 모델(v2)이 달라서" in texts(at.warning) and fake.world["approvals"][0]["status"] == "pending"


def test_503_keeps_request_pending_and_says_what_server_shows(fake):
    at = opened()
    fake.fail["approve"] = api_client.ApiError(503, "service_unavailable", "새 모델 v2 를 불러오지 못해 승인하지 않았습니다")
    click(at, page.APPROVE_BUTTON_KEY, "x")
    text = texts(at.error)
    assert "승인하지 않았습니다" in text and "요청은 승인 대기 상태" in text and "운영 모델은 v1" in text
    assert ":orange[1건]" in texts(at.markdown) and not at.button(key=page.APPROVE_BUTTON_KEY).disabled


def test_server_down_on_post_keeps_marker_and_blocks_duplicate_for_a_while(fake):
    at = opened()
    fake.fail["approve"] = api_client.ServerDown("timeout")
    click(at, page.APPROVE_BUTTON_KEY)
    assert view.SERVER_DOWN_APPROVAL_TEXT in texts(at.warning)
    assert at.session_state[state.APPROVAL_INFLIGHT_KEY]["approval_id"] == "APR-000002"
    fake.args.clear()
    fake.fail.clear()
    at.run()
    assert at.button(key=page.APPROVE_BUTTON_KEY).disabled and "버튼을 잠갔습니다" in texts(at.caption)
    assert "approve" not in fake.args  # 15초 안에는 같은 요청을 다시 보내지 않는다
    at.session_state[state.APPROVAL_INFLIGHT_KEY]["ts"] = time.time() - view.INFLIGHT_SECONDS - 1  # 15초가 지나면 무시
    at.run()
    assert not at.button(key=page.APPROVE_BUTTON_KEY).disabled


def test_marker_blocks_a_second_post_even_if_button_event_arrives(fake):
    at = opened()
    state_marker = {"approval_id": "APR-000002", "action": "approve", "ts": time.time()}
    at.session_state[state.APPROVAL_INFLIGHT_KEY] = state_marker
    at.run()
    assert at.button(key=page.APPROVE_BUTTON_KEY).disabled and "approve" not in fake.args


def test_read_server_down_keeps_last_values_grey_and_buttons_still_there(fake):
    at = opened()
    fake.fail["approvals"] = api_client.ServerDown("refused")
    at.run()
    assert not at.exception and "서버에 연결할 수 없습니다" in texts(at.error) and "기준입니다" in texts(at.caption)
    assert "APR-000002" in texts(at.markdown) and ":gray[1건]" in texts(at.markdown)
    fake.fail.clear()
    at.run()
    assert not at.error and ":orange[1건]" in texts(at.markdown)


def test_two_pending_requests_can_be_chosen_and_summary_follows_the_choice(fake):
    fake.world["approvals"] = [item(3, "v3"), item(2, "v2")]
    fake.world["runs"] = [run(result="v3", n=3, gate_result=gate(baseline="v1")), run(result="v2", n=2, gate_result=gate(baseline="v1"))]
    fake.world["models"]["v3"] = model("v3", 0.91)
    at = new_page().run()
    assert not at.exception and "후보 v3" in texts(at.markdown)
    at.selectbox(key=page.SELECT_KEY).select("APR-000002").run()
    assert not at.exception and "후보 v2" in texts(at.markdown)
    click(at, page.REJECT_BUTTON_KEY, "v2 거절")
    assert fake.args["reject"] == [{"approval_id": "APR-000002", "reason": "v2 거절"}]


def test_summary_without_run_record_says_so_and_does_not_block(fake):
    fake.world["runs"] = []
    at = new_page().run()
    assert not at.exception and "게이트 근거를 찾지 못했습니다" in texts(at.caption)
    assert not at.button(key=page.APPROVE_BUTTON_KEY).disabled and fake.args["model"] == [{"version": "v2"}]


def test_call_counts_stay_within_plan_limits(fake):
    at = new_page().run()
    assert len(fake.calls) == 5
    fake.world["models"].clear()  # 모델 상세를 못 읽는 경우에도 갱신당 상한은 같다
    fake.fail["model"] = api_client.ApiError(404, "not_found", "x")
    state_cache = at.session_state[state.RETRAIN_MODELS_KEY]
    state_cache.clear()
    fake.calls.clear()
    click(at, page.APPROVE_BUTTON_KEY)
    assert len(fake.calls) <= 6


def test_logout_clears_approval_state(fake):
    at = opened()
    click(at, page.APPROVE_BUTTON_KEY, "x")
    assert state.APPROVAL_SNAPSHOT_KEY in at.session_state and state.APPROVAL_NOTICE_KEY in at.session_state
    from streamlit.testing.v1 import AppTest

    def out():
        from frontend import state as s

        s.logout()

    check = AppTest.from_function(out)
    for key in state.APPROVAL_KEYS:
        check.session_state[key] = "x"
    check.run()
    assert not any(key in check.session_state for key in state.APPROVAL_KEYS)


def entry_with_sidebar():
    from frontend import app
    from frontend.pages import approvals as p

    app.sidebar()  # 실제 app.py 와 같은 사이드바 (운영 모델 줄은 GET /health 를 다시 읽을 때만 바뀐다)
    p.render()


def test_approve_reruns_whole_script_once_so_sidebar_shows_new_model(fake):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(entry_with_sidebar, default_timeout=15)
    at.session_state[state.AUTH_KEY] = {"token": "tok", "role": "admin", "username": "admin", "expires_at": time.time() + 3600}
    at.run()
    assert not at.exception and "v1 (불러옴)" in texts(at.sidebar.markdown)
    fake.calls.clear()
    fake.args.clear()
    at.button(key=page.APPROVE_BUTTON_KEY).click().run()
    assert not at.exception
    assert fake.calls.count("health") == 2 and len(fake.args["approve"]) == 1  # 클릭 실행 1번 + st.rerun() 1번 (되풀이되지 않음)
    assert "v2 (불러옴)" in texts(at.sidebar.markdown) and "v1 (불러옴)" not in texts(at.sidebar.markdown)
    assert "승인 완료" in texts(at.success)  # 다시 실행한 뒤에도 알림이 남는다
    fake.calls.clear()
    at.run()
    assert fake.calls.count("health") == 1 and "approve" not in fake.calls  # 그 다음 갱신은 평범한 1번
