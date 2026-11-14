"""Streamlit 화면. 백엔드 API만 호출한다 (DB·모델 파일에 직접 접근하지 않는다).

- api_client.py: HTTP 를 부르는 유일한 곳
- config.py: 서버 주소·갱신 주기·제한 시간
- state.py: 로그인 정보 보관 (st.session_state)
- view.py: 화면에 보일 문장·서식을 만드는 순수 함수 (Streamlit 없이 시험한다)
- app.py: 진입점, pages/: 화면별 파일

실행: python -m streamlit run frontend/app.py --server.port 8501
"""
