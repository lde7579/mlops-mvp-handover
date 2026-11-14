# CLAUDE.md — 자동차 부품 누락 검사 MLOps MVP

이 파일은 Claude Code가 세션 시작 때 읽는 프로젝트 규칙입니다.

## 프로젝트 요약

- 가상 데이터(시뮬레이터)로 자동차 부품 **누락(OK/NG)** 을 검사하는 이미지 분류 시스템
- 수집 → 추론 → 재학습 → **사람 승인** → 재배포가 한 바퀴 도는 MLOps MVP (대학교 프로젝트, 교수·학생 시연)
- 로컬 Windows PC, 판정 5초 이내, 프로젝트의 초점은 모델 성능이 아니라 **AI 운영**

## 사용자 정보

- 비개발자, 혼자 진행. **한국어로, 쉬운 말로** 설명할 것
- 전문 용어는 풀어서 설명하고, **왜 이렇게 하는지**를 한두 문장 덧붙일 것 (발표에서 설명해야 함)
- 에러가 나면 원인을 쉬운 말로 먼저 설명하고 해결할 것

## 기준 문서 (충돌하면 docs/가 우선)

- `docs/requirements.md` — 범위, 확정 결정, 일정
- `docs/data-contract.md` — 데이터 스키마, 모델 입출력 (파트 간 계약)
- `docs/api-v1.md` — 백엔드 API
- `docs/screens.md` — Streamlit 5개 화면
- `docs/e2e-scenario.md` — 데모 8단계 = 테스트 케이스

## 확정된 기술 스택 (변경 금지, 변경이 필요하면 이유를 설명하고 먼저 물어볼 것)

| 영역 | 확정 |
|---|---|
| 백엔드 | Python 3.11, FastAPI, SQLite(SQLAlchemy) |
| 모델 | PyTorch(CPU) 소형 CNN, scikit-learn(지표) |
| 프론트 | Streamlit (백엔드 API만 호출, DB·모델 파일 직접 접근 금지) |
| 시뮬레이터 | Python (Pillow, numpy) |
| 모델 레지스트리 | 자체 구현 (버전 테이블 + 파일 + 지표 JSON). MLflow 사용 안 함 |

## 범위 밖 (제안하지 말 것)

Kafka/MQTT, Kubernetes, Docker, MLflow, Prefect/Airflow, Grafana, GPU 학습, 누락 부품 위치 예측, 현장 연동·설비 제어(쓰기)

## 작업 규칙

1. **계획 → 승인 → 구현 → 검증** 순서. 새 작업은 먼저 계획만 보여 주고 승인을 기다린다
2. 작업 단위는 작게 (엔드포인트 1개 + 테스트 수준). 한 번에 여러 기능을 만들지 않는다
3. 계약 우선: `docs/data-contract.md`, `docs/api-v1.md`를 바꿔야 하면 코드보다 먼저 문서 변경안을 제시하고 승인받는다
4. 테스트 통과가 완료 기준. 완료 시 실행한 테스트와 결과를 알려 준다
5. 전처리는 `common/preprocess.py` 한 곳에서만 수행 (학습·추론 공유)
6. 테스트셋(`test_normal`, `test_drift`)은 고정이며 **학습에 절대 사용하지 않는다**
7. 랜덤 시드를 고정해 결과를 재현 가능하게 한다
8. 불확실한 부분은 추측하지 말고 질문하거나 가정을 명시한다
9. 파일 삭제, 데이터 초기화 같은 되돌릴 수 없는 작업은 실행 전에 확인을 받는다
10. `spike/`는 일회성 실험 코드이다. 읽기와 참고만 하고, 새 코드에서 import 하지 않는다. 필요한 부분은 복사해서 `parts/` 또는 `common/`에 계약 테스트와 함께 새로 넣는다
11. pytest 대상은 `tests/`로만 한정한다 (`pytest tests/`로 실행하고, `spike/`는 돌리지 않는다)

## 실행 환경 (Windows PowerShell)

```
cd $HOME\Documents\mlops-mvp
.\.venv\Scripts\Activate.ps1
```

설치된 패키지: numpy, pillow, scikit-learn, torch(CPU), pytest. 새 패키지가 필요하면 설치 전에 이유를 알린다.

## 폴더 구조 (목표)

```
mlops-mvp/
├── CLAUDE.md
├── docs/          # 설계 문서
├── spike/         # 드리프트 재현 스파이크 (임시, 일회성 실험, 읽기·참고만)
├── common/        # 공통 전처리/스키마
├── parts/
│   ├── collector/ # 파트1: 데이터 수집 (시뮬레이터 포함, 조원 담당)
│   └── predictor/ # 파트2: 모델 추론 (조원 담당)
├── services/
│   └── training/  # 파트3: 재학습·승격 (본인 담당)
├── backend/       # FastAPI
├── frontend/      # Streamlit
├── scripts/       # reset_demo.py 등
└── tests/         # contract/, e2e/ (pytest는 이 폴더만)
```
