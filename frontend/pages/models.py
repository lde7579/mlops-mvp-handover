"""화면 3 — 모델 관리: 버전 목록, 두 버전 지표 비교, 롤백(관리자만), 배포 이력.

원칙
- 어느 버전이 운영 중인지(stage), 지표, 배포 이력은 항상 서버(GET /models, GET /deployments)에서 다시 읽은 값만 보여 준다.
  GET /models 의 항목에는 지표가 들어 있어 모델 상세를 따로 읽지 않고 지표 캐시도 두지 않는다 (stage 와 지표가 같은 응답에서 온다).
  버튼을 누르면 같은 실행 안에서 ① POST ② 서버를 다시 읽기 ③ 그리기 순서로 처리하고, POST 응답 본문은 쓰지 않는다.
- 문장·서식은 view.py, HTTP 는 api_client.py. 이 파일은 호출 순서와 그리기만 맡는다.
- 한 번 실행의 최대 호출 수 (자동 갱신만): 2번 = 모델 목록 1 + 배포 이력 1. 버튼이 낀 실행은 POST 1번이 더해져 최대 3번.
- 롤백은 읽기 제한 시간 8초(api_client). 같은 버전의 롤백을 15초 안에 다시 보내지 않는다 (표식, view.rollback_inflight_active).
- 롤백이 서버에서 확인되면 st.rerun() 1회로 사이드바의 운영 모델도 다시 그린다 (fragment 갱신은 본문만 다시 그리므로).
"""
import time
from typing import Callable, Optional

import streamlit as st

from frontend import api_client, config, state, view

COLORS = {"ok": "green", "warn": "orange", "error": "red", "info": "blue", "muted": "gray"}
SESSION_EXPIRED = "로그인이 만료되었습니다. 다시 로그인해 주세요."
NOTICE_LEVELS = {"ok": st.success, "info": st.info, "error": st.error, "warn": st.warning, "muted": st.caption}
COMPARE_A_KEY, COMPARE_B_KEY, TARGET_KEY, REASON_KEY, CLEAR_REASON_KEY = state.MODELS_WIDGET_KEYS
ROLLBACK_BUTTON_KEY = "models_rollback"


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
def load_snapshot(client) -> dict:
    """모델 목록(지표 포함)과 배포 이력을 읽는다. ServerDown 은 그대로 던진다."""
    return {"models": _fetch(client.models), "deployments": _fetch(lambda: client.deployments(limit=view.DEPLOYMENT_LIMIT))}


# ---------- 버튼 처리 ----------
def send(client, request: dict) -> dict:
    """롤백 요청 1번. 결과: {"kind": "sent"|"down"|"blocked"|"error", "status", "message"}. 응답 본문은 쓰지 않는다."""
    version = request["version"]
    if view.rollback_inflight_active(state.models_inflight(), version, time.time()):
        return {"kind": "blocked"}
    state.set_models_inflight(version, time.time())
    try:
        client.rollback_model(version, request["reason"])
    except api_client.ApiError as e:
        _need_login(e.status)
        state.clear_models_inflight()  # 서버가 분명히 거절한 것은 바로 다시 시도할 수 있다
        return {"kind": "error", "status": e.status, "message": e.message}
    except api_client.ServerDown:  # 처리됐는지 모른다: 표식을 남겨 15초 동안 같은 요청을 다시 보내지 않는다
        return {"kind": "down"}
    return {"kind": "sent"}


# ---------- 그리기 ----------
def draw_notice() -> None:
    notice = state.models_notice()
    if notice is not None:
        NOTICE_LEVELS.get(notice["level"], st.info)(f"{notice['text']} ({view.clock_text(notice['ts'])})")


def _valid_choice(key: str, options: list, default) -> None:
    """저장된 선택이 목록에 없으면 기본값으로 되돌린다 (위젯을 만들기 전에만 바꿀 수 있다)."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = default


def draw_rollback(models_res: dict, role: str, box) -> Optional[dict]:
    """롤백 대상(archived 만) 선택, 사유 입력, [롤백]. 눌렸으면 {"version", "reason"}. 활성 여부는 직전에 읽은 값 기준(편의일 뿐, 최종 판단은 서버)."""
    targets = view.rollback_targets(models_res)
    enabled, hint = view.can_rollback(role)
    with box:
        st.markdown("**롤백 (관리자만)**")
        if not targets:
            st.caption(view.NO_ARCHIVED_TEXT)
            if hint:
                st.caption(hint)
            return None
        _valid_choice(TARGET_KEY, targets, targets[0])
        target = st.selectbox("롤백할 버전 (archived 만)", targets, key=TARGET_KEY, disabled=not enabled)
        busy = view.rollback_inflight_active(state.models_inflight(), target, time.time())
        reason = st.text_input("사유 (필수)", max_chars=500, key=REASON_KEY, disabled=not enabled)
        clicked = st.button("롤백", key=ROLLBACK_BUTTON_KEY, type="primary", disabled=not enabled or busy)
        if hint:
            st.caption(hint)
        if busy:
            st.caption("같은 롤백 요청을 방금 보냈습니다. 서버가 처리했는지 확인하는 동안(최대 15초) 버튼을 잠갔습니다.")
    return {"version": target, "reason": view.normalize_rollback_reason(reason)} if clicked else None


def draw_models(models_res: dict, stale: bool) -> None:
    if not models_res["ok"]:
        st.markdown(_colored("모델 목록을 읽지 못했습니다: " + view.md_cell(models_res["error"] or ""), "error"))
        return
    items = view.model_items(models_res)
    production = view.production_of(models_res)
    st.markdown("**모델 버전 (서버 기준)**: " + _colored(f"운영 모델 {production or '없음'}", "muted" if stale else "ok" if production else "warn"))
    if items:
        st.markdown(view.model_table_markdown(view.model_rows(items)))
    else:
        st.caption("등록된 모델 버전이 없습니다.")


def draw_compare(models_res: dict) -> None:
    items = view.model_items(models_res)
    st.markdown("**지표 비교**")
    if not items:
        return
    versions = view.model_versions(models_res)
    default_a, default_b = view.default_compare(models_res)
    if default_b is None:  # 버전이 하나뿐: 비교 대신 그 버전의 지표만
        st.caption(f"버전이 {default_a} 하나뿐이라 비교할 대상이 없습니다. 이 버전의 지표만 보여 줍니다.")
        st.markdown(view.md_table(view.SINGLE_METRIC_COLUMNS, view.single_metric_rows(items[0]["metrics"])))
        return
    _valid_choice(COMPARE_A_KEY, versions, default_a)
    _valid_choice(COMPARE_B_KEY, versions, default_b)
    col_a, col_b = st.columns(2)
    a = col_a.selectbox("버전 A (기준)", versions, key=COMPARE_A_KEY)
    b = col_b.selectbox("버전 B (비교)", versions, key=COMPARE_B_KEY)
    a_metrics, b_metrics = view.find_model(models_res, a)["metrics"], view.find_model(models_res, b)["metrics"]
    st.caption(f"기준 A = {a} ({view.stage_label(view.find_model(models_res, a)['stage'])}), 비교 B = {b} ({view.stage_label(view.find_model(models_res, b)['stage'])})")
    st.markdown(view.md_table(view.compare_columns(a, b), view.compare_rows(a, a_metrics, b, b_metrics)))
    st.markdown("혼동행렬 (양성은 NG: TN 정상을 OK, FP 정상을 NG, FN 불량을 OK, TP 불량을 NG)")
    st.markdown(view.md_table(view.compare_confusion_columns(a, b), view.compare_confusion_rows(a, a_metrics, b, b_metrics)))


def draw_deployments(deployments_res: dict) -> None:
    st.markdown("**배포 이력**")
    if not deployments_res["ok"]:
        st.markdown(_colored("배포 이력을 읽지 못했습니다: " + view.md_cell(deployments_res["error"] or ""), "error"))
        return
    data = deployments_res["data"]
    st.caption(f"전체 배포·롤백 {data['total']}건 중 최근 {len(data['items'])}건 (사유·처리자는 서버 기록)")
    if data["items"]:
        st.markdown(view.md_table(view.DEPLOYMENT_COLUMNS, view.deployment_rows(data["items"])))
    else:
        st.caption("아직 배포 기록이 없습니다.")


@st.fragment(run_every=config.refresh_seconds())
def body() -> None:
    """모델 관리 본문. 이 부분만 주기적으로 다시 그려서 사이드바와 페이지 전체는 깜빡이지 않는다."""
    info = state.auth()
    if info is None:
        st.rerun()
        return
    if st.session_state.pop(CLEAR_REASON_KEY, False):  # 처리된 롤백의 사유가 다음 입력에 남지 않게 입력칸을 비운다 (입력칸을 만들기 전에만 바꿀 수 있다)
        st.session_state[REASON_KEY] = ""
    boxes = {"list": st.container(), "compare": st.container(), "rollback": st.container(), "notice": st.container(), "history": st.container()}
    previous = state.last_models_snapshot()
    request = draw_rollback(previous["snap"]["models"], info["role"], boxes["rollback"]) if previous is not None else None
    snap, result, refresh_sidebar = None, None, False
    try:
        with state.client() as client:
            if request is not None:
                result = send(client, request)
            try:
                snap = load_snapshot(client)
            except api_client.ServerDown:
                snap = None
            if snap is not None and view.has_unauthorized(snap):
                raise SessionExpired()
    except SessionExpired:  # 세션 만료(401): 알린 뒤 로그아웃한다
        state.logout()
        st.session_state[state.LOGIN_ERROR_KEY] = SESSION_EXPIRED
        st.rerun()
        return
    except api_client.ServerDown:  # 버튼 처리 중 꺼진 경우도 이번 갱신 전체를 실패로 본다
        snap = None
    if snap is not None and view.rollback_inflight_done(state.models_inflight(), snap["models"]):
        state.clear_models_inflight()  # 서버가 처리됐다고 확인했다
    if request is not None:
        text, level = view.rollback_result(request["version"], result, None if snap is None else snap["models"])
        state.set_models_notice(text, level, time.time())
        if result["kind"] == "sent" and level == "ok":
            st.session_state[CLEAR_REASON_KEY] = True
            refresh_sidebar = True  # 운영 모델이 바뀌었으니 사이드바도 다시 그린다
    if snap is not None:
        state.set_models_snapshot(snap, time.time())
    if refresh_sidebar:
        st.rerun()  # 전체를 한 번 다시 실행한다. 알림·스냅샷은 session_state 에 있어서 그대로 보인다
        return
    last = state.last_models_snapshot()
    stale = snap is None
    if last is not None and previous is None:  # 첫 그리기: 읽은 뒤에야 입력칸을 만들 수 있다
        draw_rollback(last["snap"]["models"], info["role"], boxes["rollback"])
    with boxes["notice"]:
        if stale:
            st.error(view.server_down_text(config.api_url()))
        draw_notice()
    if last is None:
        st.caption(view.last_success_text(None, config.refresh_seconds()))
        return
    if stale:
        st.caption(f"아래 값은 {view.clock_text(last['ts'])} 기준입니다 (서버가 꺼져 있어 새로 읽지 못했습니다).")
    with boxes["list"]:
        draw_models(last["snap"]["models"], stale)
    with boxes["compare"]:
        draw_compare(last["snap"]["models"])
    with boxes["history"]:
        draw_deployments(last["snap"]["deployments"])
        st.caption(view.last_success_text(last["ts"], config.refresh_seconds()))


def render() -> None:
    st.title("모델 관리")
    info = state.auth()
    if info is None:
        st.info("왼쪽 사이드바에서 로그인해 주세요. 로그인하면 모델 버전, 지표 비교, 배포 이력을 볼 수 있습니다.")
        return
    st.caption(f"{view.session_text(info['role'], info['username'])} 로 로그인했습니다. 롤백은 관리자만 할 수 있습니다.")
    body()
