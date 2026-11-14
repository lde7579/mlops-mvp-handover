"""화면 5 — 승인: 대기 요청 목록, 변경 요약(게이트·신구 지표), 승인/거절(관리자만), 처리 이력.

원칙
- 대기인지, 처리됐는지, 지금 운영 모델이 무엇인지는 항상 서버(GET /approvals, GET /models/production)에서 다시 읽은 값만 보여 준다.
  버튼을 누르면 같은 실행 안에서 ① POST ② 서버를 다시 읽기 ③ 그리기 순서로 처리하고, POST 응답 본문은 쓰지 않는다.
- 문장·서식은 view.py, HTTP 는 api_client.py. 이 파일은 호출 순서, 모델 지표 캐시, 그리기만 맡는다.
- 한 번 실행의 최대 호출 수 (자동 갱신만): 대기 요청이 없으면 1번(승인 목록). 있으면 5번 = 승인 목록 1 + 학습 이력 1 + 운영 모델 1
  + [처음 보는 버전일 때만] 모델 상세 최대 2. 버튼이 낀 실행은 POST 1번이 더해져 최대 6번 (처리 뒤 대기가 없어져도 운영 모델은 읽는다).
- 모델 지표 캐시는 재학습 화면과 같은 것(state.retrain_models)을 쓴다. 지표만 쓰고, 캐시에 든 stage 는 낡을 수 있어 표시하지 않는다.
- 승인·거절은 읽기 제한 시간 8초(api_client). 같은 요청을 15초 안에 다시 보내지 않는다 (표식, view.inflight_active).
"""
import time
from typing import Callable, Optional

import streamlit as st

from frontend import api_client, config, state, view

COLORS = {"ok": "green", "warn": "orange", "error": "red", "info": "blue", "muted": "gray"}
SESSION_EXPIRED = "로그인이 만료되었습니다. 다시 로그인해 주세요."
NOTICE_LEVELS = {"ok": st.success, "info": st.info, "error": st.error, "warn": st.warning, "muted": st.caption}
SELECT_KEY, REASON_KEY, CLEAR_REASON_KEY = state.APPROVAL_WIDGET_KEYS
APPROVE_BUTTON_KEY, REJECT_BUTTON_KEY = "approval_approve", "approval_reject"


class SessionExpired(Exception):
    """서버가 401 로 답했다 (토큰 만료)."""


def _colored(text: str, level: str) -> str:
    return f":{COLORS[level]}[{text}]"


def _fetch(call: Callable) -> dict:
    """한 건 조회. 서버 오류 응답은 결과에 담고, 서버가 꺼진 경우(ServerDown)만 그대로 던진다."""
    try:
        return view.ok_result(call())
    except api_client.ApiError as e:
        return view.fail_result(e.status, e.message)


def _need_login(status: Optional[int]) -> None:
    if status == 401:
        raise SessionExpired()


# ---------- 조회 ----------
def _fetch_production(client) -> dict:
    """운영 모델이 아직 없으면(404) 오류가 아니라 '없음'(data=None)으로 본다."""
    result = _fetch(client.production_model)
    return view.ok_result(None) if result["status"] == 404 else result


def load_snapshot(client, after_request: bool = False) -> dict:
    """승인 목록(처리된 것 포함)을 읽는다. 대기 요청이 있을 때만 학습 이력과 운영 모델도 읽는다.
    버튼을 누른 직후(after_request)에는 대기가 없어졌더라도 운영 모델을 읽는다 (승인 결과를 서버 값으로 확인하려고). ServerDown 은 그대로 던진다."""
    snap = {"approvals": _fetch(lambda: client.approvals(limit=view.APPROVAL_LIMIT)), "runs": None, "production": None}
    pending = view.pending_items(snap["approvals"])
    if pending:
        snap["runs"] = _fetch(lambda: client.training_runs(view.HISTORY_LIMIT))
    if pending or after_request:
        snap["production"] = _fetch_production(client)
    return snap


def refresh_models(client, snap: dict, chosen_id: Optional[str]) -> None:
    """고른 요청의 변경 요약에 필요한 모델 상세 중 아직 없는 것만 받아 보관한다 (지표는 바뀌지 않는다). 실패한 것은 다음 갱신에 다시 시도한다."""
    item = view.selected_item(view.pending_items(snap["approvals"]), chosen_id)
    if item is None:
        return
    cache = state.retrain_models()
    for version in view.summary_models(item["model_version"], view.candidate_run(snap["runs"], item["model_version"])):
        if version not in cache:
            result = _fetch(lambda: client.model(version))
            _need_login(result["status"])
            if result["ok"]:
                cache[version] = result["data"]


# ---------- 버튼 처리 ----------
def send(client, request: dict) -> dict:
    """승인·거절 요청 1번. 결과: {"kind": "sent"|"down"|"blocked"|"error", "status", "message"}. 응답 본문은 쓰지 않는다."""
    approval_id, action = request["approval_id"], request["action"]
    if view.inflight_active(state.approval_inflight(), approval_id, time.time()):
        return {"kind": "blocked"}
    state.set_approval_inflight(approval_id, action, time.time())
    try:
        if action == "approve":
            client.approve_approval(approval_id, request["reason"])
        else:
            client.reject_approval(approval_id, request["reason"])
    except api_client.ApiError as e:
        _need_login(e.status)
        state.clear_approval_inflight()  # 서버가 분명히 거절한 것은 바로 다시 시도할 수 있다
        return {"kind": "error", "status": e.status, "message": e.message}
    except api_client.ServerDown:  # 처리됐는지 모른다: 표식을 남겨 15초 동안 같은 요청을 다시 보내지 않는다
        return {"kind": "down"}
    return {"kind": "sent"}


# ---------- 그리기 ----------
def draw_notice() -> None:
    notice = state.approval_notice()
    if notice is not None:
        NOTICE_LEVELS.get(notice["level"], st.info)(f"{notice['text']} ({view.clock_text(notice['ts'])})")


def draw_inputs(snap: dict, role: str, boxes: dict) -> Optional[dict]:
    """요청 선택, 사유 입력, [승인] [거절]. 눌린 버튼이 있으면 {"action", "approval_id", "reason"}. 활성 여부는 직전에 읽은 값 기준(편의일 뿐, 최종 판단은 서버)."""
    pending = view.pending_items(snap["approvals"])
    if not pending:
        return None
    ids = [i["approval_id"] for i in pending]
    with boxes["select"]:
        chosen = st.selectbox("처리할 요청", ids, format_func=lambda i: view.pending_label(view.find_approval(snap["approvals"], i)), key=SELECT_KEY) \
            if len(ids) > 1 else ids[0]
    enabled, hint = view.can_decide(role)
    busy = view.inflight_active(state.approval_inflight(), chosen, time.time())
    with boxes["decide"]:
        st.markdown(f"**결정 — 후보 {view.find_approval(snap['approvals'], chosen)['model_version']} ({chosen})**")
        reason = st.text_input("사유 (승인은 선택, 거절은 필수)", max_chars=500, key=REASON_KEY, disabled=not enabled)
        approve_col, reject_col = st.columns(2)
        approve = approve_col.button("승인", key=APPROVE_BUTTON_KEY, type="primary", disabled=not enabled or busy, width="stretch")
        reject = reject_col.button("거절", key=REJECT_BUTTON_KEY, disabled=not enabled or busy, width="stretch")
        if hint:
            st.caption(hint)
        if busy:
            st.caption("같은 요청을 방금 보냈습니다. 서버가 처리했는지 확인하는 동안(최대 15초) 버튼을 잠갔습니다.")
    action = "approve" if approve else "reject" if reject else None
    return None if action is None else {"action": action, "approval_id": chosen, "reason": view.normalize_reason(action, reason)}


def draw_pending(snap: dict, stale: bool) -> None:
    approvals = snap["approvals"]
    if not approvals["ok"]:
        st.markdown(_colored("승인 요청을 읽지 못했습니다: " + view.md_cell(approvals["error"] or ""), "error"))
        return
    pending = view.pending_items(approvals)
    st.markdown("**대기 요청 (서버 기준)**: " + _colored(f"{len(pending)}건", "muted" if stale else "warn" if pending else "ok"))
    if pending:
        st.markdown(view.md_table(view.PENDING_COLUMNS, view.pending_rows(pending)))
    else:
        st.caption("승인 대기 중인 요청이 없습니다. 재학습에서 게이트를 통과하면 여기에 나타납니다.")


def draw_summary(snap: dict, chosen_id: Optional[str]) -> None:
    item = view.selected_item(view.pending_items(snap["approvals"]), chosen_id)
    if item is None:
        return
    version = item["model_version"]
    run = view.candidate_run(snap["runs"], version)
    st.markdown(f"**변경 요약 — 후보 {version}**")
    st.caption(view.production_line(snap["production"]))
    if run is None:
        st.caption(f"게이트 근거를 찾지 못했습니다 (최근 {view.HISTORY_LIMIT}건의 학습 기록에 {version}이 없습니다). 승인 가능 여부는 서버가 판단합니다.")
    else:
        st.markdown(f"**게이트 결과** — {view.gate_basis_text(run)}")
        rows = view.gate_rows(run.get("gate_result"))
        if rows:
            st.markdown(view.gate_markdown(rows))
    warning = view.baseline_warning(run, snap["production"])
    if warning:
        st.warning(warning)
    cache = state.retrain_models()
    candidate, baseline = cache.get(version), view.run_baseline(run)
    if candidate is None:
        st.caption("후보 모델의 지표를 읽지 못했습니다 (다음 갱신에서 다시 읽습니다).")
        return
    old = cache.get(baseline)
    if baseline is not None and old is None:
        st.caption(f"기준 모델 {baseline}의 지표를 읽지 못했습니다 (다음 갱신에서 다시 읽습니다).")
    old_metrics = None if old is None else old["metrics"]
    st.markdown("**신·구 모델 지표 비교**")
    st.caption(f"기존 = {baseline or '없음'}, 후보 = {version}")
    st.markdown(view.md_table(view.METRIC_COLUMNS, view.metric_rows(old_metrics, candidate["metrics"])))
    st.markdown("혼동행렬 (양성은 NG: TN 정상을 OK, FP 정상을 NG, FN 불량을 OK, TP 불량을 NG)")
    st.markdown(view.md_table(view.CONFUSION_COLUMNS, view.confusion_rows(old_metrics, candidate["metrics"])))


def draw_history(snap: dict) -> None:
    st.markdown("**처리 이력**")
    approvals = snap["approvals"]
    if not approvals["ok"]:
        return
    handled = view.handled_items(approvals)
    st.caption(f"전체 승인 요청 {approvals['data']['total']}건 중 최근 {len(approvals['data']['items'])}건에서 처리된 {len(handled)}건 (처리자·시각·사유는 서버 기록)")
    if handled:
        st.markdown(view.md_table(view.APPROVAL_HISTORY_COLUMNS, view.approval_history_rows(handled)))
    else:
        st.caption("아직 처리된 요청이 없습니다.")


@st.fragment(run_every=config.refresh_seconds())
def body() -> None:
    """승인 본문. 이 부분만 주기적으로 다시 그려서 사이드바와 페이지 전체는 깜빡이지 않는다."""
    info = state.auth()
    if info is None:
        st.rerun()
        return
    if st.session_state.pop(CLEAR_REASON_KEY, False):  # 처리된 요청의 사유가 다음 요청에 남지 않게 입력칸을 비운다 (입력칸을 만들기 전에만 바꿀 수 있다)
        st.session_state[REASON_KEY] = ""
    boxes = {"select": st.container(), "pending": st.container(), "decide": st.container(), "notice": st.container(),
             "summary": st.container(), "history": st.container()}
    previous = state.last_approval_snapshot()
    request = draw_inputs(previous["snap"], info["role"], boxes) if previous is not None else None
    snap, result, refresh_sidebar = None, None, False
    try:
        with state.client() as client:
            if request is not None:
                result = send(client, request)
            try:
                snap = load_snapshot(client, after_request=request is not None)
            except api_client.ServerDown:
                snap = None
            if snap is not None:
                if view.has_unauthorized(snap):
                    raise SessionExpired()
                refresh_models(client, snap, st.session_state.get(SELECT_KEY))
    except SessionExpired:  # 세션 만료(401): 알린 뒤 로그아웃한다
        state.logout()
        st.session_state[state.LOGIN_ERROR_KEY] = SESSION_EXPIRED
        st.rerun()
        return
    except api_client.ServerDown:  # 모델 상세를 읽다가 꺼진 경우: 이번 갱신 전체를 실패로 본다
        snap = None
    if snap is not None and view.inflight_done(state.approval_inflight(), snap["approvals"]):
        state.clear_approval_inflight()  # 서버가 처리됐다고 확인했다
    if request is not None:
        text, level = view.action_result(request["action"], request["approval_id"], result,
                                         *(None, None, None) if snap is None else (snap["approvals"], snap["runs"], snap["production"]))
        state.set_approval_notice(text, level, time.time())
        if result["kind"] == "sent" and level == "ok":
            st.session_state[CLEAR_REASON_KEY] = True
            refresh_sidebar = request["action"] == "approve"  # 운영 모델이 바뀌었으니 사이드바도 다시 그린다
    if snap is not None:
        state.set_approval_snapshot(snap, time.time())
    if refresh_sidebar:
        st.rerun()  # 전체를 한 번 다시 실행한다. 알림·스냅샷은 session_state 에 있어서 그대로 보인다
        return
    last = state.last_approval_snapshot()
    stale = snap is None
    if last is not None and previous is None:  # 첫 그리기: 읽은 뒤에야 입력칸을 만들 수 있다
        draw_inputs(last["snap"], info["role"], boxes)
    with boxes["notice"]:
        if stale:
            st.error(view.server_down_text(config.api_url()))
        draw_notice()
    if last is None:
        st.caption(view.last_success_text(None, config.refresh_seconds()))
        return
    if stale:
        st.caption(f"아래 값은 {view.clock_text(last['ts'])} 기준입니다 (서버가 꺼져 있어 새로 읽지 못했습니다).")
    with boxes["pending"]:
        draw_pending(last["snap"], stale)
    with boxes["summary"]:
        draw_summary(last["snap"], st.session_state.get(SELECT_KEY))
    with boxes["history"]:
        draw_history(last["snap"])
        st.caption(view.last_success_text(last["ts"], config.refresh_seconds()))


def render() -> None:
    st.title("승인")
    info = state.auth()
    if info is None:
        st.info("왼쪽 사이드바에서 로그인해 주세요. 로그인하면 승인 대기 요청과 처리 이력을 볼 수 있습니다.")
        return
    st.caption(f"{view.session_text(info['role'], info['username'])} 로 로그인했습니다. 승인과 거절은 관리자만 할 수 있습니다.")
    body()
