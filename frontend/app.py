"""Streamlit 진입점: 사이드바(로그인, 운영 모델, 백엔드 상태)와 화면 이동.

실행: python -m streamlit run frontend/app.py --server.port 8501
(python -m 으로 실행해야 프로젝트 루트가 import 경로에 들어가 `from frontend import ...` 가 된다.)
"""
import streamlit as st

from frontend import api_client, config, state, view
from frontend.pages import approvals, dashboard, inspection, models, retrain

LEVEL_ICON = {"ok": ":green[●]", "warn": ":orange[●]", "error": ":red[●]"}


def read_health():
    """GET /health. 서버가 꺼졌으면 (None, 연결 실패 문장)."""
    try:
        with api_client.ApiClient() as client:
            return client.health(), None
    except api_client.ServerDown:
        return None, f"서버에 연결할 수 없습니다 ({config.api_url()})"
    except api_client.ApiError as e:
        return None, f"서버 오류 응답: {e.message}"


def login_form() -> None:
    with st.sidebar.form("login"):
        username = st.selectbox("계정", config.LOGIN_ACCOUNTS, format_func=view.role_label)
        password = st.text_input("비밀번호", type="password")
        submitted = st.form_submit_button("로그인")
    if submitted:
        try:
            with api_client.ApiClient() as client:
                response = client.login(username, password)
            state.set_login(response)
            st.rerun()
        except api_client.ServerDown:
            st.session_state[state.LOGIN_ERROR_KEY] = view.login_error_message(None, True, config.api_url())
        except api_client.ApiError as e:
            st.session_state[state.LOGIN_ERROR_KEY] = view.login_error_message(e.status, False, config.api_url())
    if state.LOGIN_ERROR_KEY in st.session_state:
        st.sidebar.error(st.session_state[state.LOGIN_ERROR_KEY])


def sidebar() -> None:
    st.sidebar.header("MLOps 검사 시스템")
    info = state.auth()
    if info is None:
        login_form()
    else:
        st.sidebar.write(f"로그인: **{view.session_text(info['role'], info['username'])}**")
        if st.sidebar.button("로그아웃"):
            state.logout()
            st.rerun()
    health, problem = read_health()
    backend_text, backend_level = view.backend_status(health)
    model_text, model_level = view.model_status(health)
    st.sidebar.divider()
    st.sidebar.write(f"백엔드 상태: {LEVEL_ICON[backend_level]} {backend_text}")
    st.sidebar.write(f"운영 모델: {LEVEL_ICON[model_level]} {model_text}")
    if problem:
        st.sidebar.caption(problem)
    st.sidebar.caption(f"서버 주소: {config.api_url()}")


def main() -> None:
    st.set_page_config(page_title="MLOps 자동차 부품 누락 검사", layout="wide")
    sidebar()
    page = st.navigation([st.Page(dashboard.render, title="대시보드", url_path="dashboard", default=True),
                          st.Page(inspection.render, title="검사/추론", url_path="inspection"),
                          st.Page(retrain.render, title="재학습", url_path="retrain"),
                          st.Page(approvals.render, title="승인", url_path="approvals"),
                          st.Page(models.render, title="모델 관리", url_path="models")])
    page.run()


if __name__ == "__main__":  # streamlit 이 실행할 때만 (시험은 main() 을 직접 부른다)
    main()
