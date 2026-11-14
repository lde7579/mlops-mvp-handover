"""화면 4 — 재학습: 실행 버튼, 4단계 진행 표시, 게이트·신구 지표 결과, 실행 이력.

원칙
- 실행 중인지, 어느 단계인지, 성공·실패·게이트 결과는 항상 GET /training/runs 의 최신 run 값만 보여 준다 (화면에는 상태를 기억하는 변수를 두지 않는다).
  화면을 떠났다 돌아와도 같은 호출로 그대로 복원된다. 버튼을 누르면 같은 실행 안에서 ① POST ② 서버를 다시 읽기 ③ 그리기 순서로 처리하고, POST 응답 본문은 쓰지 않는다.
- 문장·서식은 view.py, HTTP 는 api_client.py. 이 파일은 호출 순서, 모델 지표 캐시, 그리기만 맡는다.
- 한 번 실행의 최대 호출 수 (자동 갱신만): 5번 = 학습 이력 1 + pool 건수 1 + [최근 run 이 게이트 통과일 때만] 승인 목록 1 + [처음 보는 버전일 때만] 모델 상세 최대 2.
  버튼이 낀 실행은 POST 1번이 더해져 최대 6번. 읽기 제한 시간은 기본값(3초)을 그대로 쓴다 (실화면 점검에서 측정한 뒤에 바꿀지 정한다).
"""
import time
from typing import Callable, Optional

import streamlit as st

from frontend import api_client, config, state, view

COLORS = {"ok": "green", "warn": "orange", "error": "red", "info": "blue", "muted": "gray"}
SESSION_EXPIRED = "로그인이 만료되었습니다. 다시 로그인해 주세요."
NOTICE_LEVELS = {"ok": st.success, "info": st.info, "error": st.error, "warn": st.warning, "muted": st.caption}
START_REQUESTED_TEXT = "재학습 요청을 보냈습니다. 아래 진행 상태는 서버에서 다시 읽은 값입니다."
BUTTON_KEY = "retrain_start"


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
    """학습 이력(최근 run 포함)과 pool 건수를 읽는다. 최근 run 이 게이트를 통과했으면 승인 목록도 읽는다. ServerDown 은 그대로 던진다."""
    snap = {"runs": _fetch(lambda: client.training_runs(view.HISTORY_LIMIT)),
            "pool": _fetch(lambda: client.inspections(data_split="pool", limit=1)), "approvals": None}
    if view.needs_approval_check(view.latest_run(snap["runs"])):  # 승인 요청은 처리되면 status 가 바뀌므로 매번 다시 읽는다
        snap["approvals"] = _fetch(lambda: client.approvals(limit=view.APPROVAL_LIMIT))
    return snap


def refresh_models(client, run: Optional[dict]) -> None:
    """지표 비교에 필요한 모델 상세 중 아직 없는 것만 받아 보관한다 (지표는 바뀌지 않는다). 실패한 것은 보관하지 않아 다음 갱신에 다시 시도한다."""
    cache = state.retrain_models()
    for version in view.models_needed(run):
        if version not in cache:
            result = _fetch(lambda: client.model(version))
            _need_login(result["status"])
            if result["ok"]:
                cache[version] = result["data"]


# ---------- 버튼 처리 ----------
def run_start(client) -> Optional[str]:
    """재학습 실행 요청. 409 는 서버를 다시 읽은 뒤에야 두 가지를 구분할 수 있으므로 서버 메시지를 돌려주고, 나머지는 여기서 알림을 남긴다."""
    try:
        client.start_training_run()
    except api_client.ApiError as e:
        _need_login(e.status)
        if e.status == 409:
            return e.message
        text, level = view.start_failure_message(e.status, e.message)
    except api_client.ServerDown:
        text, level = view.SERVER_DOWN_START_TEXT, "warn"
    else:
        text, level = START_REQUESTED_TEXT, "ok"
    state.set_retrain_notice(text, level, time.time())
    return None


# ---------- 그리기 ----------
def draw_notice() -> None:
    notice = state.retrain_notice()
    if notice is not None:
        NOTICE_LEVELS.get(notice["level"], st.info)(f"{notice['text']} ({view.clock_text(notice['ts'])})")


def draw_stages(run: Optional[dict], stale: bool) -> None:
    text, level = view.run_status_text(run)
    st.markdown("**진행 상태 (서버 기준)**: " + _colored(text, "muted" if stale else level))
    for column, (name, mark, mark_level) in zip(st.columns(4), view.run_stages(run)):
        column.markdown(f"**{name}**")
        column.markdown(_colored(mark, "muted" if stale else mark_level))


def draw_metrics(run: dict) -> None:
    cache = state.retrain_models()
    version, baseline = run.get("result_version"), (run.get("gate_result") or {}).get("baseline_version")
    candidate = cache.get(version)
    st.markdown("**신·구 모델 지표 비교**")
    if candidate is None:
        st.caption("후보 모델의 지표를 읽지 못했습니다 (다음 갱신에서 다시 읽습니다).")
        return
    old = cache.get(baseline)
    if baseline is not None and old is None:
        st.caption(f"기준 모델 {baseline}의 지표를 읽지 못했습니다 (다음 갱신에서 다시 읽습니다).")
    old_metrics = None if old is None else old["metrics"]
    st.caption(f"기존 = {baseline or '없음'}, 후보 = {version}")
    st.markdown(view.md_table(view.METRIC_COLUMNS, view.metric_rows(old_metrics, candidate["metrics"])))
    st.markdown("혼동행렬 (양성은 NG: TN 정상을 OK, FP 정상을 NG, FN 불량을 OK, TP 불량을 NG)")
    st.markdown(view.md_table(view.CONFUSION_COLUMNS, view.confusion_rows(old_metrics, candidate["metrics"])))


def draw_result(run: Optional[dict], snap: dict) -> None:
    outcome = view.run_outcome(run, snap["approvals"])
    if outcome is None:
        return
    st.markdown("**결과**")
    NOTICE_LEVELS[outcome[1]](outcome[0])
    if run["status"] == "failed":
        st.code(run.get("error_message") or "오류 내용 없음", language=None)
    note = view.pool_note(run)
    if note:
        st.caption(note)
    if run["status"] == "succeeded":
        st.caption(view.snapshot_text(run))
        st.markdown(f"**게이트 결과** — {view.gate_basis_text(run)}")
        rows = view.gate_rows(run.get("gate_result"))
        if rows:
            st.markdown(view.gate_markdown(rows))
        draw_metrics(run)


def draw_history(snap: dict) -> None:
    st.markdown("**실행 이력**")
    runs = snap["runs"]
    if not runs["ok"]:
        st.caption("실행 이력을 읽지 못했습니다: " + view.md_cell(runs["error"] or ""))
        return
    items = runs["data"]["items"]
    if not items:
        st.caption("아직 재학습을 한 적이 없습니다.")
        return
    st.caption(f"전체 {runs['data']['total']}건 중 최근 {len(items)}건 (소요는 시작~종료 시각으로 화면에서 계산한 값)")
    st.markdown(view.run_history_markdown(view.run_history_rows(items)))


@st.fragment(run_every=config.refresh_seconds())
def body() -> None:
    """재학습 본문. 이 부분만 주기적으로 다시 그려서 사이드바와 페이지 전체는 깜빡이지 않는다."""
    if state.auth() is None:
        st.rerun()
        return
    previous = state.last_retrain_snapshot()
    active = previous is not None and view.is_run_active(view.latest_run(previous["snap"]["runs"]))  # 버튼 잠금은 직전 서버 상태 기준(편의일 뿐)
    st.markdown("**재학습 실행**")
    pool_box = st.container()
    clicked = st.button("재학습 실행", key=BUTTON_KEY, disabled=active, width="content")
    if active:
        st.caption("재학습이 진행 중이라 버튼을 잠갔습니다 (직전 확인 기준: 방금 끝났다면 다음 갱신에서 풀립니다).")
    notice_box = st.container()
    snap, conflict = None, None
    try:
        with state.client() as client:
            if clicked:
                conflict = run_start(client)
            try:
                snap = load_snapshot(client)
            except api_client.ServerDown:
                snap = None
            if snap is not None:
                if view.has_unauthorized(snap):
                    raise SessionExpired()
                refresh_models(client, view.latest_run(snap["runs"]))
    except SessionExpired:  # 세션 만료(401): 알린 뒤 로그아웃한다
        state.logout()
        st.session_state[state.LOGIN_ERROR_KEY] = SESSION_EXPIRED
        st.rerun()
        return
    except api_client.ServerDown:  # 모델 상세를 읽다가 꺼진 경우: 이번 갱신 전체를 실패로 본다
        snap = None
    if conflict is not None:
        runs, pool = (None, None) if snap is None else (snap["runs"], snap["pool"])
        text, level = view.conflict_message(runs, pool, conflict)
        state.set_retrain_notice(text, level, time.time())
    if snap is not None:
        state.set_retrain_snapshot(snap, time.time())
    last = state.last_retrain_snapshot()
    stale = snap is None
    with pool_box:
        if last is not None:
            pool_text, pool_level = view.pool_text(last["snap"]["pool"])
            st.markdown("pool 건수 (아직 재학습에 쓰지 않은 새 검사): " + _colored(pool_text, "muted" if stale else pool_level))
    with notice_box:
        if stale:
            st.error(view.server_down_text(config.api_url()))
        draw_notice()
    if last is None:
        st.caption(view.last_success_text(None, config.refresh_seconds()))
    else:
        if stale:
            st.caption(f"아래 값은 {view.clock_text(last['ts'])} 기준입니다 (서버가 꺼져 있어 새로 읽지 못했습니다).")
        run = view.latest_run(last["snap"]["runs"])
        draw_stages(run, stale)
        draw_result(run, last["snap"])
        draw_history(last["snap"])
        st.caption(view.last_success_text(last["ts"], config.refresh_seconds()))
    st.divider()
    st.caption("한계 ① " + view.LIMIT_POOL_TEXT)
    st.caption("한계 ② " + view.LIMIT_EVALUATE_TEXT)
    st.caption("안내 ③ " + view.LIMIT_APPROVAL_TEXT)


def render() -> None:
    st.title("재학습")
    info = state.auth()
    if info is None:
        st.info("왼쪽 사이드바에서 로그인해 주세요. 로그인하면 재학습을 실행하고 결과를 볼 수 있습니다.")
        return
    st.caption(f"{view.session_text(info['role'], info['username'])} 로 로그인했습니다. 운영자와 관리자 모두 재학습을 실행할 수 있습니다.")
    body()
