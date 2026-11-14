"""화면 2 — 검사/추론: 시뮬레이터 제어, 최신 검사(이미지·판정), 판정 이력 표.

원칙
- 시뮬레이터가 켜져 있는지, 드리프트가 켜져 있는지는 항상 GET /simulator/status 값만 보여 준다 (화면에는 상태를 기억하는 변수를 두지 않는다).
  버튼을 누르면 같은 실행 안에서 ① 그 버튼의 API 호출 ② 상태를 서버에서 다시 읽기 ③ 그리기 순서로 처리한다.
- 문장·서식은 view.py, HTTP 는 api_client.py. 이 파일은 호출 순서, 캐시, 그리기만 맡는다.
- 한 번 실행의 최대 호출 수 (자동 갱신만): 4번 = 상태 1 + 판정 이력 1 + [새 최신 검사일 때만] 검사 상세 1 + 이미지 1.
  버튼이 낀 실행은 최대 6번 (수동 1건 생성+판정은 생성·판정·이미지 3번 + 상태 1 + 이력 1 + 이미지 1, 이때 검사 상세는 읽지 않는다).
- 최신 검사의 이미지는 id 가 바뀔 때만 받는다 (session_state 에 1건만 보관).
"""
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import streamlit as st

from frontend import api_client, config, state, view

HISTORY_LIMIT = 20  # 판정 이력 표 행 수 상한
COLORS = {"ok": "green", "warn": "orange", "error": "red", "info": "blue", "muted": "gray"}
SESSION_EXPIRED = "로그인이 만료되었습니다. 다시 로그인해 주세요."
DRIFT_TYPES = ("dark", "noise", "shift")
RATE_KEY, INTERVAL_KEY, DRIFT_KEY = "inspection_defect_rate", "inspection_interval", "inspection_drift_type"
CONTINUOUS_SECONDS = 3 * config.REFRESH_SECONDS  # 이보다 오래 이 화면을 보지 않았으면 '이미 있던 판정'으로 보고 표시 지연을 재지 않는다
NOTICE_LEVELS = {"ok": st.success, "info": st.info, "error": st.error, "warn": st.warning}


class SessionExpired(Exception):
    """서버가 401 로 답했다 (토큰 만료)."""


def _now() -> datetime:
    """현재 시각(UTC). 표시 지연(다) 계산용. 시험에서 바꿔 끼운다."""
    return datetime.now(timezone.utc)


def _tick() -> float:
    """경과 시간(나) 측정용 시계. 시험에서 바꿔 끼운다."""
    return time.perf_counter()


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
    """상태와 판정 이력을 읽는다. ServerDown 이 나면 그대로 던진다 (나머지 호출은 하지 않는다)."""
    return {"simulator": _fetch(client.simulator_status), "predictions": _fetch(lambda: client.predictions(HISTORY_LIMIT))}


def refresh_latest(client, snap: dict, clicked: bool) -> Optional[dict]:
    """최신 판정의 이미지·생성 시각 캐시를 맞춘다. 같은 검사 id 면 아무것도 다시 받지 않는다.

    표시 지연(다)은 '이 화면을 계속 보고 있었고, 버튼 실행이 아닐 때' 새로 나타난 최신 판정에 대해서만 한 번 잰다.
    """
    result = snap["predictions"]
    cache = state.latest_cache()
    if not result["ok"] or not result["data"]["items"]:
        return cache
    inspection_id = result["data"]["items"][0]["inspection_id"]
    wall = time.time()
    if cache is not None and cache["inspection_id"] == inspection_id:
        state.set_latest_cache({**cache, "checked_at": wall})
        return state.latest_cache()
    seen_at = _now()  # 화면이 이 판정을 처음 읽은 시각
    measurable = cache is not None and wall - cache["checked_at"] <= CONTINUOUS_SECONDS and not clicked
    captured_at, delay = None, None
    if measurable:
        detail = _fetch(lambda: client.inspection(inspection_id))
        _need_login(detail["status"])
        if detail["ok"]:
            captured_at = detail["data"]["captured_at"]
            delay = view.seconds_between(captured_at, seen_at)
    image = _fetch(lambda: client.inspection_image(inspection_id))  # 실패(404 등)도 캐시에 남겨 같은 id 를 다시 받지 않는다
    _need_login(image["status"])
    state.set_latest_cache({"inspection_id": inspection_id, "source": "auto", "image": image["data"], "image_error": image["error"],
                            "captured_at": captured_at, "delay": delay, "checked_at": wall})
    return state.latest_cache()


# ---------- 버튼 처리 ----------
def run_manual(client) -> tuple:
    """1건 생성 -> 판정 -> 이미지 받기. 이 전체 경과가 (나) 값이다. 반환: (문장, 수준)."""
    started = _tick()
    try:
        inspection = client.simulator_generate()
    except api_client.ApiError as e:
        _need_login(e.status)
        return view.manual_failure_text(None, e.status, e.message), "error"
    inspection_id = inspection["inspection_id"]
    try:
        client.predict(inspection_id)
    except api_client.ApiError as e:
        _need_login(e.status)
        return view.manual_failure_text(inspection_id, e.status, e.message), "error"
    image, image_error = None, None
    try:
        image = client.inspection_image(inspection_id)
    except api_client.ApiError as e:
        _need_login(e.status)
        image_error = e.message
    elapsed = _tick() - started
    state.set_latest_cache({"inspection_id": inspection_id, "source": "manual", "image": image, "image_error": image_error,
                            "captured_at": inspection.get("captured_at"), "delay": None, "checked_at": time.time()})
    return view.elapsed_text(elapsed)


def run_action(client, action: str, params: dict) -> None:
    """눌린 버튼의 API 를 부르고 결과 알림을 남긴다. 상태 표시는 이후 GET /simulator/status 로 다시 읽는다 (응답 본문은 쓰지 않는다)."""
    try:
        if action == "manual":
            text, level = run_manual(client)
        elif action == "start":
            client.simulator_start(params["defect_rate"], params["interval_seconds"])
            text, level = "시작 요청을 보냈습니다. 위의 상태는 서버에서 다시 읽은 값입니다.", "ok"
        elif action == "stop":
            client.simulator_stop()
            text, level = "중지 요청을 보냈습니다. 자동 생성이 멈춘 상태입니다 (이미 멈춰 있었어도 정상입니다).", "ok"
        elif action == "drift_on":
            client.simulator_drift(True, params["drift_type"])
            text, level = "드리프트 켜기 요청을 보냈습니다. 위의 상태는 서버에서 다시 읽은 값입니다.", "ok"
        else:
            client.simulator_drift(False)
            text, level = "드리프트 끄기 요청을 보냈습니다. 위의 상태는 서버에서 다시 읽은 값입니다.", "ok"
    except api_client.ApiError as e:
        _need_login(e.status)
        text, level = view.action_message(action, e.status, e.message)
    except api_client.ServerDown:
        text, level = view.SERVER_DOWN_ACTION_TEXT, "error"
    state.set_notice(text, level, time.time())


# ---------- 그리기 ----------
def draw_controls(last_status: Optional[dict]) -> tuple:
    """제어 영역. 눌린 버튼이 있으면 (이름, 입력값)을 돌려준다. 시작/중지의 활성 여부는 직전 상태 기준(편의일 뿐, 오류는 서버 응답으로 처리)."""
    running = None if last_status is None else last_status["running"]
    if running:  # 실행 중에는 입력칸에도 서버 값을 보여 준다
        st.session_state[RATE_KEY], st.session_state[INTERVAL_KEY] = float(last_status["defect_rate"]), float(last_status["interval_seconds"])
    st.session_state.setdefault(RATE_KEY, 0.2)
    st.session_state.setdefault(INTERVAL_KEY, 2.0)
    st.session_state.setdefault(DRIFT_KEY, "dark")
    action, params = None, {}
    first, second = st.columns(2)
    rate = first.number_input("불량률 (0~1)", min_value=0.0, max_value=1.0, step=0.05, format="%.2f", key=RATE_KEY, disabled=bool(running))
    interval = second.number_input("생성 간격 (초, 1~3)", min_value=1.0, max_value=3.0, step=0.5, format="%.1f", key=INTERVAL_KEY, disabled=bool(running))
    start_col, stop_col = st.columns(2)
    if start_col.button("시작", key="inspection_start", disabled=running is True, width="stretch"):
        action, params = "start", {"defect_rate": round(float(rate), 2), "interval_seconds": float(interval)}
    if stop_col.button("중지", key="inspection_stop", disabled=running is False, width="stretch"):
        action = "stop"
    if st.button("1건 생성+판정 (정상)", key="inspection_manual", width="stretch"):
        action = "manual"
    drift_type = st.selectbox("드리프트 종류", DRIFT_TYPES, format_func=lambda t: view.DRIFT_LABELS[t], key=DRIFT_KEY)
    on_col, off_col = st.columns(2)
    if on_col.button("드리프트 켜기", key="inspection_drift_on", width="stretch"):
        action, params = "drift_on", {"drift_type": drift_type}
    if off_col.button("드리프트 끄기", key="inspection_drift_off", width="stretch"):
        action = "drift_off"
    return action, params


def draw_status(last: Optional[dict], stale: bool) -> None:
    st.markdown("**시뮬레이터 상태 (서버 기준)**")
    if last is None:
        st.caption("아직 상태를 읽지 못했습니다.")
        return
    result = last["snap"]["simulator"]
    if not result["ok"]:
        st.markdown(_colored("상태를 읽지 못했습니다: " + (result["error"] or ""), "error"))
        return
    text, level = view.simulator_detail_text(result["data"])
    st.markdown(_colored(text, "muted" if stale else level))


def draw_notice() -> None:
    notice = state.notice()
    if notice is not None:
        NOTICE_LEVELS.get(notice["level"], st.info)(f"{notice['text']} ({view.clock_text(notice['ts'])})")


def draw_latest(last: Optional[dict], entry: Optional[dict], stale: bool) -> None:
    st.markdown("**최신 검사**")
    if last is None or not last["snap"]["predictions"]["ok"]:
        st.caption("판정 이력을 읽지 못했습니다.")
        return
    items = last["snap"]["predictions"]["data"]["items"]
    if not items:
        st.caption("아직 판정이 없습니다. [시작] 또는 [1건 생성+판정]을 눌러 보세요.")
        return
    newest = items[0]
    level_of = (lambda level: "muted" if stale else level)
    st.markdown(f"검사 {newest['inspection_id']} · {view.local_time_text(newest['predicted_at'])} 판정")
    mine = entry if entry is not None and entry["inspection_id"] == newest["inspection_id"] else None
    if mine is not None and mine["image"]:
        st.image(mine["image"], width=256, caption="검사 이미지 (128×128)")
    elif mine is not None:
        st.caption("이미지를 불러올 수 없습니다" + (f" ({mine['image_error']})" if mine["image_error"] else ""))
    ng = newest["predicted_label"] == "NG"
    st.markdown(f"# {_colored(newest['predicted_label'], level_of('error' if ng else 'ok'))}")
    st.markdown(f"점수 {newest['score']:.2f} (기준 {newest['threshold']:.2f}) · 모델 {newest['model_version']}")
    latency, latency_level = view.latency_text(newest["latency_ms"])
    st.markdown(_colored(latency, level_of(latency_level)))
    kind = view.misjudgement_kind(newest["true_label"], newest["predicted_label"])
    st.markdown(_colored(view.match_detail(newest["true_label"], kind), level_of("ok" if kind is None else "error")))
    if mine is not None and mine["source"] == "auto":
        st.caption(view.display_delay_text(mine["delay"]))


def draw_history(last: Optional[dict]) -> None:
    st.markdown("**판정 이력**")
    if last is None or not last["snap"]["predictions"]["ok"]:
        return
    data = last["snap"]["predictions"]["data"]
    st.caption(view.ng_summary_text(data["items"]) + f" · 전체 {data['total']}건 중 최근 {len(data['items'])}건")
    if data["items"]:
        st.markdown(view.history_markdown(view.history_rows(data["items"])))


@st.fragment(run_every=config.refresh_seconds())
def body() -> None:
    """검사/추론 본문. 이 부분만 주기적으로 다시 그려서 사이드바와 페이지 전체는 깜빡이지 않는다."""
    if state.auth() is None:
        st.rerun()
        return
    previous = state.last_inspection_snapshot()
    last_status = previous["snap"]["simulator"]["data"] if previous is not None and previous["snap"]["simulator"]["ok"] else None
    left, right = st.columns([2, 3])
    with left:
        status_box = st.container()
        action, params = draw_controls(last_status)
        notice_box = st.container()
    snap, entry = None, None
    try:
        with state.client() as client:
            if action is not None:
                run_action(client, action, params)
            try:
                snap = load_snapshot(client)
            except api_client.ServerDown:
                snap = None
            if snap is not None:
                if view.has_unauthorized(snap):
                    raise SessionExpired()
                entry = refresh_latest(client, snap, clicked=action is not None)
    except SessionExpired:  # 세션 만료(401): 알린 뒤 로그아웃한다
        state.logout()
        st.session_state[state.LOGIN_ERROR_KEY] = SESSION_EXPIRED
        st.rerun()
        return
    except api_client.ServerDown:  # 이미지·상세를 읽다가 꺼진 경우: 이번 갱신 전체를 실패로 본다
        snap = None
    if snap is not None:
        state.set_inspection_snapshot(snap, time.time())
    last = state.last_inspection_snapshot()
    stale = snap is None
    with status_box:
        if stale:
            st.error(view.server_down_text(config.api_url()))
        draw_status(last, stale)
    with notice_box:
        draw_notice()
    with right:
        draw_latest(last, entry if entry is not None else state.latest_cache(), stale)
    if stale and last is not None:
        st.caption(f"아래 값은 {view.clock_text(last['ts'])} 기준입니다 (서버가 꺼져 있어 새로 읽지 못했습니다).")
    draw_history(last)
    st.caption(view.last_success_text(None if last is None else last["ts"], config.refresh_seconds()))


def render() -> None:
    st.title("검사/추론")
    info = state.auth()
    if info is None:
        st.info("왼쪽 사이드바에서 로그인해 주세요. 로그인하면 시뮬레이터를 켜고 판정을 볼 수 있습니다.")
        return
    st.caption(f"{view.session_text(info['role'], info['username'])} 로 로그인했습니다.")
    body()
