"""화면 1 — 대시보드 본문: 단계 상태 줄, 카드 6개, 성능 경고, 자동 갱신.

- 호출은 load_snapshot 한 곳에서 모아서 하고, 화면은 그 결과(snapshot)만 그린다 (문장·서식은 view.py).
- 서버가 꺼지면(ServerDown) 남은 호출을 건너뛰고, 마지막 성공 값을 시각과 함께 흐리게 보여 준다.
- 한 번 갱신의 최대 호출 수는 11번이다 (API 이름은 load_snapshot 참고): 상태 확인 1, 시뮬레이터 1, 학습 이력 1, 운영 모델 1,
  검사 건수 2(오늘, pool), 판정 통계 1, 성능 1, 승인 대기 1, 드리프트 경고 1, 배포 기록 1(판단 보류일 때만).
"""
import time
from datetime import datetime
from typing import Callable, Optional

import streamlit as st

from frontend import api_client, config, state, view

COLORS = {"ok": "green", "warn": "orange", "error": "red", "info": "blue", "muted": "gray"}
SESSION_EXPIRED = "로그인이 만료되었습니다. 다시 로그인해 주세요."


def _fetch(call: Callable) -> dict:
    """한 건 조회. 서버 오류 응답은 결과에 담고, 서버가 꺼진 경우(ServerDown)만 그대로 던진다."""
    try:
        return view.ok_result(call())
    except api_client.ApiError as e:
        return view.fail_result(e.status, e.message)


def _fetch_production(client) -> dict:
    result = _fetch(client.production_model)
    if not result["ok"] and result["status"] == 404:  # 운영 모델 없음은 오류가 아니라 하나의 상태다
        return view.ok_result(None)
    return result


def load_snapshot(client, now: datetime) -> dict:
    """대시보드에 필요한 조회를 모두 한다. ServerDown 이 나면 그대로 던진다 (나머지 호출은 하지 않는다)."""
    snap = {
        "health": _fetch(client.health),
        "simulator": _fetch(client.simulator_status),
        "training": _fetch(client.training_runs),
        "production": _fetch_production(client),
        "today": _fetch(lambda: client.inspections(since=view.today_start(now))),
        "pool": _fetch(lambda: client.inspections(data_split="pool")),
        "stats": _fetch(client.prediction_stats),
        "performance": _fetch(client.performance),
        "approvals": _fetch(lambda: client.approvals(status="pending")),
        "drift": _fetch(client.drift),
        "deployments": None,
    }
    if view.needs_deployment_check(snap["drift"]):
        snap["deployments"] = _fetch(client.deployments)
    return snap


def _colored(text: str, level: str) -> str:
    return f":{COLORS[level]}[{text}]"


def draw_stages(snap: dict, stale: bool) -> None:
    for column, (name, text, level) in zip(st.columns(4), view.stages(snap)):
        column.markdown(f"**{name}**")
        column.markdown(_colored(text, "muted" if stale else level))


def draw_cards(snap: dict, stale: bool) -> None:
    for column, card in zip(st.columns(6), view.cards(snap)):
        column.markdown(f"**{card['label']}**")
        column.markdown(f"### {_colored(card['value'], 'muted' if stale else card['level'])}")
        if card["caption"]:
            column.caption(card["caption"])


def draw_warning(snap: dict, stale: bool) -> None:
    text, level = view.performance_warning(snap["drift"], snap["deployments"])
    if stale or level == "muted":
        with st.container(border=True):
            st.markdown(_colored(text, "muted"))
    elif level == "error":
        st.error(text)
    else:
        st.success(text)


@st.fragment(run_every=config.refresh_seconds())
def body() -> None:
    """대시보드 본문. 이 부분만 주기적으로 다시 그려서 사이드바와 페이지 전체는 깜빡이지 않는다."""
    if state.auth() is None:  # 토큰이 만료됐으면 로그인 화면으로 돌아간다
        st.rerun()
        return
    snap: Optional[dict] = None
    try:
        with state.client() as client:
            snap = load_snapshot(client, datetime.now().astimezone())
    except api_client.ServerDown:
        st.error(view.server_down_text(config.api_url()))
    if snap is not None:
        if view.has_unauthorized(snap):  # 세션 만료(401): 카드마다 알리지 않고 한 번만 알린 뒤 로그아웃한다
            state.logout()
            st.session_state[state.LOGIN_ERROR_KEY] = SESSION_EXPIRED
            st.rerun()
            return
        state.set_snapshot(snap, time.time())
    last = state.last_snapshot()
    if last is None:
        st.caption(view.last_success_text(None, config.refresh_seconds()))
        return
    stale = snap is None
    if stale:
        st.caption(f"아래 값은 {view.clock_text(last['ts'])} 기준입니다 (서버가 꺼져 있어 새로 읽지 못했습니다).")
    draw_stages(last["snap"], stale)
    st.divider()
    draw_cards(last["snap"], stale)
    draw_warning(last["snap"], stale)
    st.caption(view.last_success_text(last["ts"], config.refresh_seconds()))


def render() -> None:
    st.title("대시보드")
    info = state.auth()
    if info is None:
        st.info("왼쪽 사이드바에서 로그인해 주세요. 로그인하면 이 화면에 운영 상태가 나타납니다.")
        return
    st.caption(f"{view.session_text(info['role'], info['username'])} 로 로그인했습니다.")
    body()
