"""로그인 정보 보관. st.session_state 에만 둔다 (디스크·브라우저 저장소에는 두지 않는다. 새로고침하면 다시 로그인한다)."""
import time
from typing import Optional

import streamlit as st

from frontend import api_client, view

AUTH_KEY = "auth"
LOGIN_ERROR_KEY = "login_error"
INSPECTION_SNAPSHOT_KEY = "inspection_snapshot"  # 검사/추론 화면의 마지막 성공 값 {"snap", "ts"}
INSPECTION_CACHE_KEY = "inspection_latest"  # 최신 검사 1건의 이미지·생성 시각·표시 지연 (같은 id 는 다시 받지 않는다)
INSPECTION_NOTICE_KEY = "inspection_notice"  # 마지막 버튼 결과 {"text", "level", "ts"}
INSPECTION_KEYS = (INSPECTION_SNAPSHOT_KEY, INSPECTION_CACHE_KEY, INSPECTION_NOTICE_KEY)
RETRAIN_SNAPSHOT_KEY = "retrain_snapshot"  # 재학습 화면의 마지막 성공 값 {"snap", "ts"}
RETRAIN_MODELS_KEY = "retrain_models"  # 버전 -> GET /models/{version} 응답 (지표는 바뀌지 않으므로 한 번만 받는다)
RETRAIN_NOTICE_KEY = "retrain_notice"  # 마지막 버튼 결과 {"text", "level", "ts"}
RETRAIN_KEYS = (RETRAIN_SNAPSHOT_KEY, RETRAIN_MODELS_KEY, RETRAIN_NOTICE_KEY)
APPROVAL_SNAPSHOT_KEY = "approval_snapshot"  # 승인 화면의 마지막 성공 값 {"snap", "ts"}
APPROVAL_NOTICE_KEY = "approval_notice"  # 마지막 버튼 결과 {"text", "level", "ts"}
APPROVAL_INFLIGHT_KEY = "approval_inflight"  # 보낸 승인·거절 요청 표식 {"approval_id", "action", "ts"} (중복 전송 방지용, 15초 뒤 무시)
APPROVAL_WIDGET_KEYS = ("approval_select", "approval_reason", "approval_clear_reason")  # 선택·사유 입력칸 (화면 값일 뿐 서버 상태가 아님)
APPROVAL_KEYS = (APPROVAL_SNAPSHOT_KEY, APPROVAL_NOTICE_KEY, APPROVAL_INFLIGHT_KEY) + APPROVAL_WIDGET_KEYS
MODELS_SNAPSHOT_KEY = "models_snapshot"  # 모델 관리 화면의 마지막 성공 값 {"snap", "ts"}
MODELS_NOTICE_KEY = "models_notice"  # 마지막 버튼 결과 {"text", "level", "ts"}
MODELS_INFLIGHT_KEY = "models_inflight"  # 보낸 롤백 요청 표식 {"version", "ts"} (중복 전송 방지용, 15초 뒤 무시)
MODELS_WIDGET_KEYS = ("models_compare_a", "models_compare_b", "models_rollback_target", "models_rollback_reason", "models_clear_reason")  # 화면 값일 뿐 서버 상태가 아님
MODELS_KEYS = (MODELS_SNAPSHOT_KEY, MODELS_NOTICE_KEY, MODELS_INFLIGHT_KEY) + MODELS_WIDGET_KEYS


def set_login(response: dict, now: Optional[float] = None) -> None:
    """POST /auth/login 응답을 보관한다."""
    now = time.time() if now is None else now
    st.session_state[AUTH_KEY] = {
        "token": response["access_token"], "role": response["role"], "username": response["username"],
        "expires_at": now + float(response["expires_in"])}
    st.session_state.pop(LOGIN_ERROR_KEY, None)


def logout() -> None:
    st.session_state.pop(AUTH_KEY, None)
    st.session_state.pop("dashboard_snapshot", None)  # 로그아웃하면 마지막 성공 값도 지운다 (SNAPSHOT_KEY)
    for key in INSPECTION_KEYS + RETRAIN_KEYS + APPROVAL_KEYS + MODELS_KEYS:  # 검사/추론·재학습·승인·모델 관리 화면의 마지막 성공 값·캐시·알림도 함께 지운다
        st.session_state.pop(key, None)


def auth() -> Optional[dict]:
    """로그인 정보. 없거나 만료됐으면 None (만료된 정보는 지운다)."""
    info = st.session_state.get(AUTH_KEY)
    if info is None:
        return None
    if view.is_expired(info.get("expires_at"), time.time()):
        logout()
        return None
    return info


def is_logged_in() -> bool:
    return auth() is not None


def is_admin() -> bool:
    info = auth()
    return info is not None and info.get("role") == "admin"


def client() -> api_client.ApiClient:
    """현재 로그인 정보(토큰)로 만든 API 클라이언트. 로그인 전이면 토큰 없이 만든다. `with state.client() as c:` 로 쓰면 연결을 닫는다."""
    info = auth()
    return api_client.ApiClient(token=None if info is None else info["token"])


# ---------- 대시보드 마지막 성공 값 (서버가 꺼졌을 때 시각과 함께 흐리게 보여 주기 위해) ----------
SNAPSHOT_KEY = "dashboard_snapshot"


def set_snapshot(snap: dict, ts: float) -> None:
    st.session_state[SNAPSHOT_KEY] = {"snap": snap, "ts": ts}


def last_snapshot() -> Optional[dict]:
    """{"snap", "ts"} 또는 None."""
    return st.session_state.get(SNAPSHOT_KEY)


# ---------- 검사/추론 화면 (④-2) ----------
def set_inspection_snapshot(snap: dict, ts: float) -> None:
    st.session_state[INSPECTION_SNAPSHOT_KEY] = {"snap": snap, "ts": ts}


def last_inspection_snapshot() -> Optional[dict]:
    return st.session_state.get(INSPECTION_SNAPSHOT_KEY)


def set_latest_cache(entry: dict) -> None:
    st.session_state[INSPECTION_CACHE_KEY] = entry


def latest_cache() -> Optional[dict]:
    return st.session_state.get(INSPECTION_CACHE_KEY)


def set_notice(text: str, level: str, ts: float) -> None:
    st.session_state[INSPECTION_NOTICE_KEY] = {"text": text, "level": level, "ts": ts}


def notice() -> Optional[dict]:
    return st.session_state.get(INSPECTION_NOTICE_KEY)


# ---------- 재학습 화면 (④-3) ----------
def set_retrain_snapshot(snap: dict, ts: float) -> None:
    st.session_state[RETRAIN_SNAPSHOT_KEY] = {"snap": snap, "ts": ts}


def last_retrain_snapshot() -> Optional[dict]:
    return st.session_state.get(RETRAIN_SNAPSHOT_KEY)


def retrain_models() -> dict:
    """버전 -> 모델 상세 응답. 같은 dict 를 돌려주므로 호출한 쪽이 채우면 보관된다."""
    return st.session_state.setdefault(RETRAIN_MODELS_KEY, {})


def set_retrain_notice(text: str, level: str, ts: float) -> None:
    st.session_state[RETRAIN_NOTICE_KEY] = {"text": text, "level": level, "ts": ts}


def retrain_notice() -> Optional[dict]:
    return st.session_state.get(RETRAIN_NOTICE_KEY)


# ---------- 승인 화면 (④-4) ----------
def set_approval_snapshot(snap: dict, ts: float) -> None:
    st.session_state[APPROVAL_SNAPSHOT_KEY] = {"snap": snap, "ts": ts}


def last_approval_snapshot() -> Optional[dict]:
    return st.session_state.get(APPROVAL_SNAPSHOT_KEY)


def set_approval_notice(text: str, level: str, ts: float) -> None:
    st.session_state[APPROVAL_NOTICE_KEY] = {"text": text, "level": level, "ts": ts}


def approval_notice() -> Optional[dict]:
    return st.session_state.get(APPROVAL_NOTICE_KEY)


def set_approval_inflight(approval_id: str, action: str, ts: float) -> None:
    st.session_state[APPROVAL_INFLIGHT_KEY] = {"approval_id": approval_id, "action": action, "ts": ts}


def approval_inflight() -> Optional[dict]:
    return st.session_state.get(APPROVAL_INFLIGHT_KEY)


def clear_approval_inflight() -> None:
    st.session_state.pop(APPROVAL_INFLIGHT_KEY, None)


# ---------- 모델 관리 화면 (④-5) ----------
def set_models_snapshot(snap: dict, ts: float) -> None:
    st.session_state[MODELS_SNAPSHOT_KEY] = {"snap": snap, "ts": ts}


def last_models_snapshot() -> Optional[dict]:
    return st.session_state.get(MODELS_SNAPSHOT_KEY)


def set_models_notice(text: str, level: str, ts: float) -> None:
    st.session_state[MODELS_NOTICE_KEY] = {"text": text, "level": level, "ts": ts}


def models_notice() -> Optional[dict]:
    return st.session_state.get(MODELS_NOTICE_KEY)


def set_models_inflight(version: str, ts: float) -> None:
    st.session_state[MODELS_INFLIGHT_KEY] = {"version": version, "ts": ts}


def models_inflight() -> Optional[dict]:
    return st.session_state.get(MODELS_INFLIGHT_KEY)


def clear_models_inflight() -> None:
    st.session_state.pop(MODELS_INFLIGHT_KEY, None)
