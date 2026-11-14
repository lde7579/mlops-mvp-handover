# 백엔드 API v1 — MVP 최소 세트

> 기준: `mlops-mvp-project-brief.md` 8장 + `data-contract.md`
> 구현은 FastAPI. **OpenAPI 스펙(`/openapi.json`)이 최종 계약**이며, 이 문서는 사람이 읽기 위한 요약입니다.
> Streamlit 프론트는 DB·모델 파일에 직접 접근하지 않고 아래 API만 호출합니다.

---

## 0. 기획서 대비 추가 제안 (승인 필요)

| # | 추가 | 이유 | 영향 범위 |
|---|---|---|---|
| A1 | `POST /auth/login` | 기획서는 "단순 JWT 또는 고정 계정"이라고만 적었고, 로그인 엔드포인트가 목록에 없음. 고정 계정 2개로 토큰을 발급하는 최소 기능 | 인증 모듈, 프론트 로그인 화면(사이드바) |
| A2 | `GET /health` 응답 형식 명시 | 사이드바 "백엔드 상태"와 대시보드 "단계 상태"(`screens.md` 13·36행)가 이 값을 사용함 | Streamlit 대시보드, 추론부 연결 구현(`loaded`) |
| A3 | `POST /auth/login` 요청·응답·실패 형식 명시 | 프론트 로그인과 토큰 보관(`st.session_state`)이 이 형식에 의존 | Streamlit 사이드바 로그인, 이후 모든 보호 엔드포인트 |
| A4 | 오류 응답의 `error` 문자열 6개(+ 405) 명시, 입력 검증 실패는 422가 아니라 400, 401·405 표준 헤더 | 문서에 `error` 값이 없었고 FastAPI 기본(422)이 문서 코드표와 달랐음 | 모든 엔드포인트의 오류 처리, 프론트 오류 표시 |
| A5 | `GET /predictions` 쿼리·응답 형식 명시 (필터 `model_version`·`inspection_id`·`wrong`, `{"items","total"}`, 항목에 `true_label`·`correct` 추가) | 프론트 판정 이력 표(화면 2)가 정답·정오를 한 번에 필요로 하고, 계약 3장은 정오를 저장하지 않고 조회 때 계산하도록 했음 | Streamlit 검사/추론 화면, 대시보드 |
| A6 | `GET /inspections` 쿼리·응답 형식 명시 (`{"items", "total"}`, `limit` 1~1000, 기간 필터 규칙) | 프론트 검사 목록·페이지 넘기기가 `total`과 기간 경계 규칙에 의존함 | Streamlit 검사 목록·대시보드 |
| A7 | `POST /simulator/generate` 요청·응답 명시 (201, 검사 8개 필드, `data_split`은 항상 pool) | 프론트 "수동 생성" 버튼이 이 형식에 의존함 | Streamlit 시뮬레이터 화면 |
| A8 | `GET /inspections/{id}`·`/image` 응답 명시 (판정 목록 `predictions`, PNG, 이미지 오류 404) | 프론트 상세 화면과 이미지 표시가 이 형식에 의존함 | Streamlit 검사 상세 화면 |
| A9 | 공통 규칙 추가: 요청 본문의 모르는 필드는 400, API 응답 시각은 UTC `Z` 표기 (A3 `/auth/login` 요청에도 적용) | 오타를 조용히 무시하지 않기 위해, 그리고 화면마다 시각 표기가 달라지지 않게 하기 위해 | 모든 쓰기 엔드포인트, 프론트 시각 표시 |
| A10 | `POST /predict` 요청·응답 명시 (201, prediction 8개 필드, 같은 검사를 여러 번 판정 가능, 오류 404·500·503) | 프론트 판정 이력과 이후 자동 판정이 이 형식에 의존함 | Streamlit 검사/추론 화면, 시뮬레이터 자동 판정(4주차) |
| A11 | `GET /predictions/stats` 쿼리·응답 형식 명시 (`ng_rate`는 판정 기준, `recent` 정확도·재현율, 계산할 수 없으면 null) | 대시보드의 불량률·최근 정확도 카드가 이 값에 의존하고, 불량률의 기준을 못 박기 위해 | Streamlit 대시보드 |
| A12 | `GET /models`, `/models/{version}`, `/models/production` 형식 명시 (계약 5장 7개 필드, 목록은 최신 먼저, 시각은 UTC `Z`) | 모델 관리 화면의 버전 표·지표 비교와 대시보드 운영 모델 카드가 이 형식에 의존함 | Streamlit 모델 관리 화면, 대시보드 |
| A13 | `GET /health`의 `model.loaded` 의미 명시 | 대시보드·사이드바의 "추론" 단계 표시가 이 값의 의미에 의존함 | Streamlit 사이드바·대시보드 |
| A14 | 오류 코드 503 `service_unavailable` 추가 (운영 모델이 없음 / 불러오지 못함) | 클라이언트 오류도 서버 오류도 아닌 "지금은 판정할 수 없는 상태"를 화면이 구분해 보여 주기 위해 | `/predict`, 프론트 오류 표시 |
| A15 | `GET /training/runs/{id}`의 상태·진행 단계·게이트 판정 값 명시 (status는 영문 `queued`/`running`/`succeeded`/`failed`, 진행 단계는 `step`, 게이트 판정은 `gate_result`. 근거: data-contract.md C7) | 기획서는 상태를 한글로만 적었고, 계약 C7이 저장·응답 값을 영문으로 정하고 진행 단계·게이트 판정 컬럼을 추가했음 | 학습 API(4주차), Streamlit 재학습 화면(한글 표시 이름으로 변환) |
| A16 | `/training/runs` 3개의 요청·응답 명시 (POST는 201과 `queued` 한 건, 4-B에서는 trigger `manual`만 허용, 409 두 가지, 목록 `{"items","total"}`, 항목은 계약 7장 9개 필드에 `train_data_snapshot`을 더한 10개). 4장에 서버 재시작 때 학습 기록 정리 규칙 추가 | 재학습 화면의 진행 상태·게이트 수치·학습 데이터 요약(screens.md 4장)이 이 형식에 의존하고, 누적 건수·성능 임계 자동 트리거는 판단하는 코드가 없어 값만 받으면 오해를 낳기 때문 | 학습 API(4주차), Streamlit 재학습 화면 |
| A17 | `GET /approvals` 쿼리·응답 형식 명시 (필터 `status`, `{"items","total"}`, 항목은 계약 6장 8개 필드) | 승인 화면의 대기 목록과 처리 이력(screens.md 6장)이 이 형식에 의존함 | 승인 API(4주차), Streamlit 승인·대시보드 화면 |
| A18 | `POST /approvals/{id}/approve` 요청·응답·오류 명시 (선택 사유, 200과 `reload` 결과, 404·409·503). 4장 "승인 → 배포", "핫리로드 실패" 순서 수정과 "기준 모델 검사", "승인 중 판정", "교체 실패 복구" 규칙 추가 | 모델을 읽는 동안 DB 쓰기 잠금을 쥐지 않고, 읽기에 실패해도 DB가 바뀌지 않게 하며, 낡은 게이트 근거로 승인하는 것을 막기 위해 | 승인 API, 추론부 핫리로드, Streamlit 승인 화면 |
| A19 | `POST /approvals/{id}/reject` 요청·응답·오류 명시 (사유 필수, 200, 400·404·409) | 승인 화면의 거절 버튼이 이 형식에 의존하고 e2e X4·X5의 응답을 못 박기 위해 | 승인 API, Streamlit 승인 화면 |
| A20 | `POST /models/{version}/rollback` 요청·응답·오류 명시 (사유 필수, 200과 `reload` 결과, 400·404·409·503, 모델 파일이 없을 때) | 모델 관리 화면의 롤백 버튼이 이 형식에 의존하고 e2e X2·X5의 응답을 못 박기 위해 | 배포 API, Streamlit 모델 관리 화면 |
| A21 | `GET /deployments` 쿼리·응답 형식 명시 (필터 `version`·`action`, 항목 6개 필드) | 모델 관리 화면의 배포 이력 표(screens.md 4장)가 이 형식에 의존함 | 배포 API, Streamlit 모델 관리 화면 |
| A22 | `GET /audit-logs` 쿼리·응답 형식 명시 (admin 전용, 필터 `action`, 항목 6개 필드) | 승인·롤백·실패 기록(계약 7장 C9)을 관리자가 조회하는 형식을 정하기 위해 | 감사 API, Streamlit 승인 화면 |
| A23 | `POST /simulator/start`·`stop`·`drift`의 요청·응답·오류 명시, 새 `GET /simulator/status` (상태 형식 9개 필드) | 시뮬레이터 제어(screens.md 3장)와 대시보드 "수집: 동작중"(screens.md 2장)이 이 형식에 의존하는데, 문서에 자동 생성의 상태 조회가 없었음 | 시뮬레이터 API, Streamlit 검사/추론 화면, 대시보드 |
| A24 | `GET /monitoring/performance` 쿼리·응답 형식 명시 (구간별 추이 `points`, 최근 N건 `recent`, 권장 수치와 확정 여부 `criteria`) | 성능 추이 차트와 정확도 카드가 이 형식에 의존하고, N·T는 5주차 측정 전이라 "확정 전"을 응답에 담아야 함 | 모니터링 API, Streamlit 대시보드, 모델 관리 화면 |
| A25 | `GET /monitoring/drift` 응답 형식 명시 (성능 저하 경고 `performance_warning`, 입력 분포 `input_distribution`) | 성능 저하 경고 배너와 재학습 권장 배너(screens.md 2·5장)가 이 형식에 의존함 | 모니터링 API, Streamlit 대시보드, 재학습 화면 |
| A26 | 4장 규칙 추가: 자동 생성의 자동 판정, 판정 실패 시 건너뜀, 연속 생성 실패 정지, 서버 종료 때 스레드 정리, 드리프트 상태는 메모리, 모니터링 조회 기준 | 자동 판정을 누가 하는지, 실패했을 때 어떻게 되는지가 문서에 없었음 (A10 영향 범위에는 "시뮬레이터 자동 판정(4주차)"만 있음) | 시뮬레이터, 추론, 서버 종료 처리, 모니터링 |
| A27 | `criteria`(A24)와 `performance_warning`(A25)의 N·T 확정: N=30, 정확도 0.95, `confirmed`=true. 응답 형식과 필드는 그대로이고 `confirmed` 값만 true로 바뀜. 개정 규칙의 결과가 이전 권장 수치와 같음 | 5주차 측정으로 N·T를 확정함 (`requirements.md` D-3) | 모니터링 API, Streamlit 대시보드 |

**A14 확장 (4-C):** 503 `service_unavailable`은 "운영 모델이 없음 / 불러오지 못함"에 더해 **승인·롤백 때 새 모델을 불러오지 못함**(기존 운영 모델을 계속 사용)도 뜻한다 (A18, A20).

**상태: A1 승인됨 (2026-10-17), A2~A14 승인됨 (2026-10-18), A15 승인됨 (2026-10-25), A16 승인됨 (2026-10-25), A17~A22 및 A14 확장 승인됨 (2026-10-25), A23~A26 승인됨 (2026-10-25), A27 승인됨 (2026-10-31).** 그 외 엔드포인트는 기획서 8장과 동일합니다.

---

## 1. 공통 규칙

| 항목 | 규칙 |
|---|---|
| 형식 | JSON (이미지 엔드포인트만 PNG 바이너리) |
| 인증 | `Authorization: Bearer <토큰>`. 고정 계정 2개 (operator, admin) |
| 시각 | ISO 8601. API 응답의 시각은 UTC이고 `Z`로 끝난다 (예: `2026-10-18T08:51:12.415155Z`). 요청의 시각은 시간대(`Z` 또는 `+09:00` 같은 오프셋)가 있어야 한다 |
| 목록 조회 | `limit`(기본 50, 1~1000), `offset`(0 이상) 쿼리 지원. 범위를 벗어나면 400 |
| 오류 응답 | `{"error": "<아래 문자열>", "message": "설명"}` |
| 오류 코드 | 400 bad_request(잘못된 요청·입력 검증 실패), 401 unauthorized(미인증), 403 forbidden(권한 없음), 404 not_found(없음), 405 method_not_allowed(허용되지 않는 메서드), 409 conflict(상태 충돌, 예: 이미 처리된 승인), 500 server_error(서버 오류), 503 service_unavailable(지금은 처리할 수 없음: 운영 모델이 없거나 불러오지 못함, 승인·롤백 때 새 모델을 불러오지 못함) |
| 오류 판정 순서 | 401 → 403 → 400 → 404 → 409 → 503 (A18) |
| 입력 검증 실패 | 422가 아니라 400으로 응답한다 |
| 모르는 필드 | 요청 본문에 문서에 없는 필드가 있으면 400으로 응답한다 (오타를 조용히 무시하지 않는다) |
| 401 응답 헤더 | `WWW-Authenticate: Bearer` 헤더를 포함한다 |
| 405 응답 헤더 | `Allow` 헤더로 허용된 메서드를 포함한다 |
| 갱신 | 프론트는 3~5초 폴링 (SSE 사용 안 함) |

## 2. 권한

| 역할 | 할 수 있는 일 |
|---|---|
| operator (운영자) | 조회 전체, 시뮬레이터 제어, 판정, 재학습 실행 |
| admin (관리자) | 운영자 권한 전부 + **승인/거절, 롤백** |

---

## 3. 엔드포인트

### 3-1. 인증 (A1 승인 시)

| 메서드 | 경로 | 설명 | 권한 |
|---|---|---|---|
| POST | `/auth/login` | 요청 `{"username", "password"}` (이 두 항목 외의 필드가 있으면 400) → 성공 200 `{"access_token", "token_type": "bearer", "expires_in"(초), "role"("operator" 또는 "admin"), "username"}`, 계정 또는 비밀번호가 틀리면 401. 이후 요청은 `Authorization: Bearer <access_token>`으로 호출하고, 토큰 유효기간은 서버 설정(기본 8시간)이다 | 공개 |

### 3-2. 시뮬레이터 (파트1)

| 메서드 | 경로 | 설명 | 주요 입력 | 권한 |
|---|---|---|---|---|
| POST | `/simulator/start` | 자동 생성 시작. 요청 `{"defect_rate": 0.0~1.0(기본 0.2), "interval_seconds": 1.0~3.0(기본 2.0)}`이고 본문이 없거나 항목을 생략하면 기본값이다 (범위 밖 값, 숫자가 아닌 값, 문서에 없는 필드는 400). 성공은 **200**이고 응답은 아래 상태 형식이다. 이미 실행 중이면 409(메시지 "이미 실행 중입니다"). 운영 모델이 없어도 시작되며 그때는 판정만 건너뛴다 (4장). 켜면 서버가 `interval_seconds`마다 검사 1건을 만들고 바로 판정한다 (A23) | defect_rate, interval_seconds | operator+ |
| POST | `/simulator/stop` | 자동 생성 중지. 요청 본문은 없다. 성공은 **200**이고 응답은 아래 상태 형식이다. 스레드가 끝난 뒤에 응답한다(최대 5초 기다림). 실행 중이 아니어도 200이다(멱등) (A23) | — | operator+ |
| POST | `/simulator/generate` | 검사 1건 수동 생성. 요청 `{"scenario": "normal" 또는 "drift"}`이고 본문이 없거나 `scenario`를 생략하면 normal. 성공은 **201**이고 응답은 검사 8개 필드(`inspection_id`, `captured_at`, `image_path`, `true_label`, `missing_parts`, `scenario`, `data_split`, `drift_type`). 새 검사의 `data_split`은 항상 `pool`이다 | scenario(normal/drift) | operator+ |
| POST | `/simulator/drift` | 드리프트 주입 on/off. 요청 `{"enabled": true 또는 false, "drift_type": "dark", "noise", "shift" 중 하나}`. `enabled`가 true면 `drift_type`이 필수이고, false인데 `drift_type`이 있으면 400이다 (문서에 없는 필드도 400). 성공은 **200**이고 응답은 아래 상태 형식이다. 켜져 있는 동안 **자동 생성**이 `scenario`=drift와 선택한 `drift_type`으로 검사를 만들고, 끄면 normal로 돌아간다. 강도는 `collector.toml`로 정해진다. 수동 `generate`는 이 설정과 무관하게 요청의 `scenario`를 따른다. 자동 생성이 꺼져 있어도 호출할 수 있고 다음 start부터 적용된다. 상태는 서버 메모리에만 있다 (4장) (A23) | enabled, drift_type(dark/noise/shift) | operator+ |
| GET | `/simulator/status` | 자동 생성 상태. 응답은 아래 상태 형식이다 (A23) | — | 로그인 |

**상태 형식 (A23):** 9개 필드 `running`(실행 중 여부), `interval_seconds`, `defect_rate`, `drift`(`{"enabled", "drift_type"}`, 꺼져 있으면 `drift_type`은 null), `generated`(만든 검사 수), `judged`(판정까지 저장한 수), `judge_failures`(판정을 건너뛴 수), `last_error`(마지막 오류 메시지, 없으면 null), `started_at`(시작 시각, UTC `Z`). `generated`·`judged`·`judge_failures`는 start마다 0에서 다시 센다. 실행 중이 아니면 `interval_seconds`·`defect_rate`·`started_at`은 null이다 (연속 생성 실패로 스스로 멈춘 뒤에도 `last_error`는 남는다).

### 3-3. 검사

| 메서드 | 경로 | 설명 | 권한 |
|---|---|---|---|
| GET | `/inspections` | 검사 목록, 최신 먼저. 쿼리: `scenario`, `data_split`, `since`, `until`(기간. 시간대 필수, `since`는 포함·`until`은 미포함, `since`가 `until`보다 늦으면 400), `limit`, `offset`. URL의 `+`는 `%2B`로 보낸다. 응답 `{"items": [검사 8개 필드…], "total": 필터를 적용한 전체 건수}` (`total`은 `limit`·`offset`과 무관) | 로그인 |
| GET | `/inspections/{id}` | 검사 상세. 응답은 검사 8개 필드(이미지 경로, 정답 포함)에 `predictions`(판정 8개 필드의 목록, 최신 먼저, 판정 전이면 `[]`)를 더한 형식. 없는 ID는 404 | 로그인 |
| GET | `/inspections/{id}/image` | 이미지(PNG, `image/png`). 없는 ID, 없는 파일, 허용되지 않는 경로(`data/` 밖, `..`, 링크 등)는 이유를 알리지 않고 모두 같은 404 | 로그인 |

### 3-4. 추론 (파트2)

| 메서드 | 경로 | 설명 | 권한 |
|---|---|---|---|
| POST | `/predict` | inspection_id로 판정 실행 → prediction 저장 후 반환 (5초 이내). 요청 `{"inspection_id": "INS-000001"}` (이 항목 외의 필드가 있으면 400). 성공은 **201**이고 응답은 prediction 8개 필드(`prediction_id`, `inspection_id`, `model_version`, `predicted_label`, `score`, `threshold`, `latency_ms`, `predicted_at`, 시각은 UTC `Z`). `threshold`는 판정에 쓴 모델 버전의 `metrics.json` 값이고 `latency_ms`는 추론 소요 시간이다. 같은 검사를 여러 번 판정할 수 있고 판정마다 새 `prediction_id`가 생긴다. 오류: 없는 검사는 404, 이미지가 없거나 허용되지 않는 경로면 404, 저장된 이미지가 규격(128×128 흑백)에 맞지 않으면 500, 운영 모델이 없으면 503(메시지 "운영 중인 모델이 없습니다"), 운영 모델을 불러오지 못하면 503(메시지 "운영 모델을 불러오지 못했습니다"). 503은 요청할 때마다 다시 시도하므로 파일을 고치면 서버를 다시 켜지 않아도 복구된다 | operator+ |
| GET | `/predictions` | 판정 이력, 최신 먼저(`predicted_at` 내림차순, 같으면 `prediction_id` 내림차순). 쿼리: `model_version`, `inspection_id`, `wrong`(소문자 `true`는 오판만, `false`는 정판만, 생략하면 모두, 그 외 값은 400), `limit`, `offset`. 응답 `{"items": [prediction 8개 필드 + `true_label`(검사의 정답) + `correct`(`predicted_label` == `true_label`)…], "total": 필터를 적용한 전체 건수}` (`total`은 `limit`·`offset`과 무관). 없는 `model_version`·`inspection_id`는 빈 목록 | 로그인 |
| GET | `/predictions/stats` | 판정 통계. 쿼리: `recent`(최근 N건, 1~1000, 기본 100), `model_version`(선택). 응답 `{"total": 판정 전체 건수, "ng_count": 판정이 NG인 건수, "ng_rate": ng_count/total (**판정 기준** 불량률, 0건이면 null), "recent": {"n": 실제로 센 건수(최대 N), "accuracy", "recall", "confusion_matrix": [[TN, FP], [FN, TP]]}}`. `recent`는 최신 판정 N건(목록과 같은 정렬)을 검사의 `true_label`(정답)과 비교해 계산하며 양성은 NG. 계산할 수 없으면 null(`n`이 0이면 `accuracy`·`recall`이 null, 정답 NG가 없으면 `recall`이 null). `model_version`을 주면 `total`, `ng_count`, `recent` 모두 그 버전만 센다 | 로그인 |

### 3-5. 모델

| 메서드 | 경로 | 설명 | 권한 |
|---|---|---|---|
| GET | `/models` | 버전 목록, 최신 먼저(생성 시각 내림차순). 응답 `{"items": [모델 버전…], "total": N}`이고 항목은 `/models/{version}`과 같은 7개 필드. 쿼리: `stage`(선택, `candidate`/`staging`/`production`/`archived`/`rejected` 외의 값은 400) | 로그인 |
| GET | `/models/{version}` | 버전 상세. 계약 5장 7개 필드(`version`, `stage`, `metrics`, `train_data_snapshot`, `created_at`, `approved_by`, `approved_at`, 승인 전에는 `approved_by`·`approved_at`이 null). **`metrics`는 `metrics.json` 내용이며, API 응답의 시각은 UTC `Z`로 표기하므로 파일에 적힌 표기와 다를 수 있다** (변환 대상은 `created_at`, `approved_at`, `metrics.created_at`). `v숫자` 형식이 아니거나 없는 버전은 404 | 로그인 |
| GET | `/models/production` | 현재 운영 중인 모델 (형식은 `/models/{version}`과 같음). 없으면 404 `not_found`(메시지 "운영 중인 모델이 없습니다")이고, `/predict`의 503 `service_unavailable`과 메시지가 같아도 상태 코드와 `error` 이름으로 구분된다 | 로그인 |

### 3-6. 학습 (파트3)

| 메서드 | 경로 | 설명 | 권한 |
|---|---|---|---|
| POST | `/training/runs` | 재학습 실행 (백그라운드). 요청 `{"trigger": "manual"}` (본문이 없거나 `trigger`를 생략해도 manual. 이 항목 외의 필드와 `manual`이 아닌 값(`count`, `performance` 포함)은 400. 자동 트리거는 아직 없다). 성공은 **201**이고 `status`가 `queued`인 실행 한 건(아래 항목 형식)을 돌려준다. 학습은 응답 뒤에 이어서 진행된다. 오류: `queued` 또는 `running`인 학습이 이미 있으면 409(메시지 "이미 실행 중인 재학습이 있습니다"), 재학습에 쓸 신규 데이터(`data_split=pool`)가 0건이면 409(메시지 "재학습에 쓸 신규 데이터가 없습니다"), 고정 데이터셋이 없으면 503 (A16) | operator+ |
| GET | `/training/runs` | 실행 이력, 최신 먼저(`run_id` 내림차순). 쿼리: `status`(`queued`/`running`/`succeeded`/`failed` 외의 값은 400), `limit`, `offset`. 응답 `{"items": [아래 항목 형식…], "total": 필터를 적용한 전체 건수}` (`total`은 `limit`·`offset`과 무관) (A16) | 로그인 |
| GET | `/training/runs/{id}` | 실행 상세. 응답은 아래 항목 형식. 없는 ID이거나 `RUN-숫자` 형식이 아니면 404 (A15, A16) | 로그인 |

**항목 형식 (A15, A16):** 계약 7장 training_run 9개 필드(`run_id`, `status`, `trigger`, `started_at`, `finished_at`, `result_version`, `step`, `error_message`, `gate_result`)에 `train_data_snapshot`(`result_version` 모델 버전의 값, 결과 버전이 없으면 null)을 더한 10개. `status`는 영문(`queued`/`running`/`succeeded`/`failed`, 화면 표시는 대기/실행/성공/실패), `step`은 `data`/`train`/`evaluate`/`gate` 또는 null, 시각은 UTC `Z`이며 `started_at`·`finished_at`은 아직 일어나지 않았으면 null이다. 게이트 미통과도 `succeeded`이고 통과 여부는 `gate_result.passed`가 담당한다(값 규칙은 data-contract.md 7장).

### 3-7. 승인

| 메서드 | 경로 | 설명 | 권한 |
|---|---|---|---|
| GET | `/approvals` | 승인 요청 목록, 최신 먼저(`created_at` 내림차순, 같으면 `approval_id` 내림차순). 쿼리: `status`(`pending`/`approved`/`rejected` 외의 값은 400), `limit`, `offset`. 응답 `{"items": [승인 요청 8개 필드…], "total": 필터를 적용한 전체 건수}` (`total`은 `limit`·`offset`과 무관). 항목은 계약 6장 8개 필드(`approval_id`, `model_version`, `requested_by`, `status`, `reason`, `handled_by`, `handled_at`, `created_at`, 처리 전에는 `reason`·`handled_by`·`handled_at`이 null, 시각은 UTC `Z`) (A17) | 로그인 |
| POST | `/approvals/{id}/approve` | 승인 → 후보 버전 production, 기존 production archived. 요청 `{"reason": 선택}` (본문이 없어도 되고, 있으면 공백만으로 된 값, 500자를 넘는 값, 이 항목 외의 필드는 400). 성공은 **200**이고 응답은 `{"approval": 승인 요청 8개 필드, "deployment": 배포 기록 6개 필드, "new_version": "v2", "old_version": "v1"(내려간 이전 운영 버전, 없었으면 null), "reload": {"loaded_version": "v2", "elapsed_ms": 새 모델을 읽는 데 걸린 시간}}`. **성공 응답(200)은 커밋과 추론부 교체가 모두 끝난 뒤에 돌려준다** (응답을 받은 시점에 `/health`의 `production_version`과 `/predict`가 새 버전을 쓴다). 오류: `APR-숫자` 형식이 아니거나 없는 ID는 404, 이미 처리된 요청·후보가 staging이 아님·기준 모델이 바뀜(후보의 `gate_result.baseline_version`이 지금 production과 다름, 4장)은 409, 새 모델을 불러오지 못하면 503(메시지 "새 모델 vN 을 불러오지 못해 승인하지 않았습니다. 기존 운영 모델 vM 을 계속 사용합니다". 이때 승인 요청은 pending으로 남고 감사 로그에 `approve_fail`을 기록한다) (A18) | **admin** |
| POST | `/approvals/{id}/reject` | 거절 (사유 필수). 요청 `{"reason": 필수}` (빠지거나 공백만이거나 500자를 넘으면 400, 이 항목 외의 필드도 400). 성공은 **200**이고 응답은 `{"approval": 승인 요청 8개 필드, "version": "v2", "stage": "rejected"}`. 오류: 400, 없는 ID는 404, 이미 처리된 요청·후보가 staging이 아님은 409. 추론부 모델은 바뀌지 않는다 (A19) | **admin** |

### 3-8. 배포

| 메서드 | 경로 | 설명 | 권한 |
|---|---|---|---|
| POST | `/models/{version}/rollback` | archived 버전을 production으로 복귀 (사유 필수). 요청 `{"reason": 필수}` (빠지거나 공백만이거나 500자를 넘으면 400, 이 항목 외의 필드도 400). 성공은 **200**이고 응답은 `{"deployment": 배포 기록 6개 필드, "new_version", "old_version"(내려간 이전 운영 버전, 없었으면 null), "reload": {"loaded_version", "elapsed_ms"}}`. **성공 응답(200)은 커밋과 추론부 교체가 모두 끝난 뒤에 돌려준다** (응답을 받은 시점에 `/health`의 `production_version`과 `/predict`가 새 버전을 쓴다). 오류: `v숫자` 형식이 아니거나 없는 버전은 404, archived가 아닌 버전은 409, 대상 버전의 모델 파일이 없거나 불러오지 못하면 503(메시지 "모델 vN 을 불러오지 못해 롤백하지 않았습니다. 기존 운영 모델 vM 을 계속 사용합니다". DB는 바뀌지 않고 감사 로그에 `rollback_fail`을 기록한다) (A20) | **admin** |
| GET | `/deployments` | 배포·롤백 이력, 최신 먼저(`at` 내림차순, 같으면 `deployment_id` 내림차순). 쿼리: `version`, `action`(`deploy`/`rollback` 외의 값은 400), `limit`, `offset`. 응답 `{"items": [배포 기록…], "total": 필터를 적용한 전체 건수}` (`total`은 `limit`·`offset`과 무관). 항목은 6개 필드(`deployment_id`, `version`, `action`, `reason`, `actor`, `at`, 승인 때 사유를 안 냈으면 `reason`은 null, 시각은 UTC `Z`) (A21) | 로그인 |

### 3-9. 모니터링

| 메서드 | 경로 | 설명 | 권한 |
|---|---|---|---|
| GET | `/monitoring/performance` | 시간대별 정확도·재현율 추이와 최근 N건 정확도. 쿼리: `since`, `until`(기간. 시간대 필수, `since`는 포함·`until`은 미포함, `since`가 `until`보다 늦거나 같으면 400. 기본값은 `until`=지금, `since`=1시간 전), `bucket_seconds`(10~3600, 기본 30. 구간 수가 720을 넘으면 400), `recent`(최근 N건, 1~1000, 기본 30), `model_version`(선택). 응답 `{"since", "until", "bucket_seconds", "criteria": {"recent_n", "accuracy_threshold", "confirmed"}, "recent": {"model_version", "n", "accuracy", "recall", "confusion_matrix": [[TN, FP], [FN, TP]]}, "points": [{"bucket_start", "model_version", "n", "accuracy", "recall"}…]}`. 시각은 UTC `Z`이다 (A24) | 로그인 |
| GET | `/monitoring/drift` | 성능 저하 경고와 입력 분포 변화. 쿼리: `recent`(최근 N건, 1~1000, 기본 30), `model_version`(선택, 생략하면 현재 production 버전). 응답 `{"performance_warning": {"model_version", "active", "n", "accuracy", "recent_n", "accuracy_threshold", "confirmed"}, "input_distribution": {"window_n", "baseline": {"source", "n", "brightness": {"mean", "std"}, "noise": {"mean", "std"}}, "current": {"brightness", "noise"}, "distance": {"brightness_sigma", "noise_sigma"}, "note"}}` (A25) | 로그인 |

**모니터링 값 규칙 (A24, A25):**
- 기준 대상: `recent`와 `performance_warning`은 `model_version`을 주면 그 버전의 판정만, 생략하면 **현재 production 버전의 판정만** 센다. production이 없으면 `model_version`은 null이고 `n`은 0이다. 승인·롤백 직후에는 새 production 버전의 판정만 보므로 이전 모델의 판정 때문에 경고가 남지 않는다. `points`는 `model_version`을 주면 그 버전만, 생략하면 모든 버전을 (구간, 모델 버전)마다 한 줄씩 담는다
- 정확도 계산: 판정을 검사의 `true_label`(정답)과 짝지어(판정 ↔ 검사 ID) 센다. 같은 검사를 여러 번 판정했으면 판정마다 1건이다. 양성은 NG이며 `/predictions/stats`와 같은 방식이다. `recent`는 최신 판정 N건(`/predictions`와 같은 정렬)이다. 계산할 수 없으면 null이다 (`n`이 0이면 `accuracy`·`recall`이 null, 정답 NG가 없으면 `recall`이 null)
- 구간(`points`): `predicted_at` 기준이고 `bucket_start`는 UTC 시각을 `bucket_seconds` 단위로 내린 값이다 (UTC `Z`). 판정이 없는 구간은 점을 만들지 않는다
- `data_split`으로 거르지 않는다. 재학습으로 pool이 train으로 바뀌어도 같은 기간의 수치는 바뀌지 않는다
- `criteria`: N(`recent_n`)과 `accuracy_threshold`는 5주차 측정으로 **확정한 값(N=30, 정확도 0.95)** 이며 `confirmed`는 true다 (A27, `requirements.md` D-3). 확정 전에는 같은 값이 권장 수치였고 `confirmed`가 false였다. 값은 바뀌지 않았다
- `performance_warning.active`: 그 버전의 최신 판정 N건 정확도가 `accuracy_threshold` 미만이면 true, 이상이면 false다. 판정이 N건 미만이거나 production이 없으면 null(판단 보류)이다. `n`·`accuracy`는 센 판정 건수와 그 정확도, `recent_n`은 사용한 N이다. 재학습을 시작하지 않는다
- `input_distribution`: 보조 신호이고 **값만** 돌려준다 (경고 규칙은 5주차). 지표는 평균 밝기와 노이즈 정도(이웃한 가로 픽셀 값 차이의 평균)이다. `window_n`은 위 N건 판정의 검사 이미지 수이고 `current`는 그 이미지들의 평균이다 (`window_n`이 0이면 null). `baseline`은 고정 정상 학습셋(`data/dataset/train`)에서 서버가 계산한 값이며 `source`는 출처, `n`은 계산에 쓴 이미지 수, `mean`·`std`는 이미지별 값의 평균과 표준편차다. 학습셋이 없으면 `baseline`이 null이다. `distance`는 (`current` − `baseline`의 `mean`) ÷ `baseline`의 `std`이다 (`brightness_sigma`, `noise_sigma`). `baseline`이나 `current`가 null이거나 `std`가 0이면 null이다. 재학습으로 모델이 바뀌어도 `baseline`은 바뀌지 않는다. `note`는 shift(위치 이동)가 이 두 지표로 구별되지 않는다는 설명 문장이다

### 3-10. 시스템

| 메서드 | 경로 | 설명 | 권한 |
|---|---|---|---|
| GET | `/health` | 백엔드·DB·운영 모델 로딩 상태. 응답 `{"status": "ok" 또는 "degraded", "database": "ok" 또는 "error", "model": {"production_version": "v1" 또는 null, "loaded": true 또는 false}}`. `loaded`는 추론부가 `production_version`을 실제로 불러와 판정에 쓸 수 있으면 true이고, 운영 모델이 없거나 아직 불러오지 않았거나 불러오기에 실패했으면 false이다. 서버가 켜질 때 운영 모델을 미리 불러오고, `/health`는 모델 파일을 읽지 않는다. DB 오류여도 HTTP 200이고 오류 상세는 노출하지 않는다 | 공개 |
| GET | `/audit-logs` | 감사 로그 (승인·롤백·배포), 최신 먼저(`at` 내림차순, 같으면 `log_id` 내림차순). 쿼리: `action`(계약 7장 action 값 7개 외의 값은 400), `limit`, `offset`. 응답 `{"items": [감사 기록…], "total": 필터를 적용한 전체 건수}` (`total`은 `limit`·`offset`과 무관). 항목은 6개 필드(`log_id`, `actor`, `role`, `action`, `target`, `at`, 시각은 UTC `Z`) (A22) | admin |

---

## 4. 핵심 동작 규칙

| 규칙 | 내용 |
|---|---|
| 재학습 → 게이트 | 학습 성공 후 서버가 자동으로 게이트를 평가. 통과하면 `staging` + `approval_request(pending)` 자동 생성, 미통과면 `rejected` |
| 승인 → 배포 (A18) | `approve`·`rollback` 호출 시 순서는 ① 읽기 전용으로 확인(404, 409, 기준 모델 검사. 이미 처리된 요청을 위해 모델을 읽지 않는다) → ② 새 모델을 별도로 읽어 둠 → ③ 쓰기 트랜잭션에서 다시 확인하고 stage 변경·deployment 기록·audit_log 기록을 한 트랜잭션으로 커밋 → ④ 커밋이 성공한 뒤 추론부 모델을 교체(핫리로드). 모델을 읽는 동안 DB 쓰기 잠금을 쥐지 않는다. 승인·거절·롤백은 한 번에 하나씩만 실행한다 |
| 핫리로드 실패 (A18, A20) | 새 모델을 읽는 데 실패하면 아무것도 커밋하지 않는다. 기존 production을 유지하고, 승인 요청은 pending으로 남기며, 감사 로그에 `approve_fail`/`rollback_fail`(계약 C9)을 기록하고 503으로 응답한다. 실패 기록은 승인·롤백 트랜잭션과 별개로 저장한다 |
| 교체 실패 복구 (A18) | 커밋은 됐는데 교체가 실행되지 못해도, 다음 `/predict`가 DB의 production 기준으로 새 모델을 불러온다 |
| 기준 모델 검사 (A18) | 후보의 학습 기록(`result_version`이 그 후보)에 있는 `gate_result.baseline_version`이 지금 production과 다르면 승인은 409이다 (둘 다 없으면 같다고 본다. 학습 기록이 없으면 검사하지 않는다). 거절은 막지 않고, 롤백에는 적용하지 않는다 |
| 승인 중 판정 (A18) | 판정에 기록되는 모델 버전과 임계값은 항상 실제로 쓴 모델과 일치한다. 승인 직전에 시작한 판정은 v1, 직후에 시작한 판정은 v2로 기록되고, 저장 시각이 승인 시각과 몇 ms 엇갈릴 수 있다 |
| 중복 처리 방지 | 이미 처리된 승인 요청을 다시 처리하면 409 |
| 동시 학습 방지 | 실행 중인 학습이 있으면 새 학습 요청은 409 |
| 재시작 정리 | 서버가 켜질 때 `queued`·`running`으로 남은 학습은 `failed`(`error_message`: "서버가 재시작되어 중단되었습니다")로 바꾼다. 만들다 만 `models/<version>/` 폴더는 지우지 않는다 (A16) |
| 사유 필수 | 거절, 롤백은 사유가 없으면 400 |
| 자동 판정 (A26) | 자동 생성은 검사 1건을 만든 직후 같은 스레드가 판정까지 한다 (`/predict`와 같은 규칙으로 prediction 저장). 운영 모델이 없거나 불러오지 못하면 검사는 남기고 판정만 건너뛴다. 건너뛴 검사를 나중에 채우지 않으며 건너뛴 횟수는 `judge_failures`, 이유는 `last_error`에 남는다 |
| 연속 생성 실패 (A26) | 검사 생성이 연속 5번 실패하면 자동 생성이 스스로 멈춘다 (`running`=false, `last_error` 유지). 판정 실패는 세지 않는다 |
| 서버 종료와 재시작 (A26) | 서버가 꺼질 때 자동 생성 스레드를 멈추고(최대 5초 기다림) DB 연결을 닫는다. 서버를 다시 켜면 자동 생성은 멈춘 상태이고 드리프트 주입은 꺼진 상태다 (상태는 서버 메모리에만 있다. 프로세스 하나(워커 1개)만 지원한다) |
| 모니터링 조회 (A26) | 정답은 검사의 `true_label`이다. `data_split`으로 거르지 않는다. N=30·정확도 0.95는 5주차 측정으로 확정한 값이다 (A27, 3-9) |

## 5. 구현 순서 제안 (계약 우선)

| 순서 | 대상 | 이유 |
|---|---|---|
| 1 | `/health`, 시뮬레이터, 검사 | 데이터가 쌓여야 이후 기능을 확인 가능 |
| 2 | `/predict`, `/predictions`, 모델 조회 | 판정이 DB에 쌓이는 시점이 3주차(11/2~11/8) 목표 |
| 3 | 학습, 승인, 배포, 롤백 | 4주차(11/9~11/15) 목표 |
| 4 | 모니터링, 감사 로그, 인증 | 마지막에 얹는다 (초기 개발 중에는 인증 없이 시작 가능) |
