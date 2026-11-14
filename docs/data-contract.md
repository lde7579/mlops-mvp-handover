# 데이터 계약서 (Data Contract) v1.4 (동결)

> 동결일: 2026-10-18 (v1.0·v1.1, git 태그 `contract-v1.1`), v1.2·v1.3·v1.4 승인일: 2026-10-25
> 변경 이력: **v1.4** = C9 반영 (0장 다섯 번째 표, audit_log action 값 `approve_fail`·`rollback_fail` 추가). **v1.3** = C8 반영 (0장 네 번째 표, training_run 시각 필드와 pool 편입 규칙). **v1.2** = C7 반영 (0장 세 번째 표, 파트 3 보조 테이블). **v1.1** = C3~C6 반영 (0장 두 번째 표). **v1.0** (2026-10-18, 태그 `contract-v1.0`) = 최초 동결, C1·C2 포함.
> 기준 문서: 원본 기획서(저장소 외부)를 바탕으로 하되, 확정된 결정은 `docs/requirements.md`가 우선합니다.
> 이 문서는 파트 간 약속입니다. **파트 내부 구현은 자유롭게 바꿀 수 있지만, 이 문서의 입출력 형식은 합의 없이 바꾸지 않습니다.**
> 데이터 수집·모델 파트를 개선하는 분은 이 문서와 계약 테스트(`tests/contract/`)를 기준으로 작업합니다.
> **동결 규칙:** v1.0 이후 입출력 형식을 바꾸려면 아래 0장 변경 제안 표에 먼저 올리고 승인받는다. 조원 인수(11/1) 이후에는 조원에게 알린다.

---

## 0. 기획서 대비 변경 제안 (승인 필요)

| # | 변경 | 기획서 | 제안 | 이유 | 영향 범위 |
|---|---|---|---|---|---|
| C1 | `data_split` 값 확장 | train / pool / test | train / pool / **test_normal / test_drift** | 테스트셋을 정상용·드리프트용으로 분리해(각 500건: 정상 400건 + 불량 100건) "하락 → 회복"을 각각 측정 | inspection 테이블, 재학습 평가 로직, 모니터링 차트 |
| C2 | 승격 게이트 조건 변경 | 재현율 ≥ 기존 +1%p, 정확도 ≥ 기존 −1%p | G1 test_drift 정확도 ≥ 기존 +1%p, G2 test_drift 재현율 ≥ 기존 −1%p, G3 test_normal 정확도 ≥ 기존 −1%p (5장 참조) | 스파이크(`spike/results.md`)에서 드리프트가 재현율 하락이 아니라 오경보 증가(정확도 하락)로 나타남. 기존 기준이면 회복된 v2도 재현율 100%→100%라 통과하지 못함 | 승격 게이트 평가 로직, 재학습 화면의 게이트 결과 표시, requirements.md·screens.md·e2e-scenario.md |

**상태: C1, C2 승인됨 (2026-10-17).** 본 문서의 아래 스키마가 그대로 유효합니다.

### v1.0 이후 변경 제안 (v1.1)

| # | 변경 | v1.0 | 제안 | 이유 | 영향 범위 |
|---|---|---|---|---|---|
| C3 | 학습 진입점 입력 형식 | 학습 데이터 목록(inspection_id 목록)을 받는다 | 학습 이미지 배열(N, 128, 128, uint8)과 라벨 목록을 받는다. inspection_id → 이미지 변환은 파트 3 또는 백엔드가 맡는다. 평가용 이미지·라벨(normal, drift)과 버전 이름(`v` + 숫자, 이미 있는 버전은 덮어쓰지 않음)도 함께 받는다 (상세: `parts/predictor/README.md`) | 파트 2가 파일·DB 위치를 몰라도 되고, 평가 이미지와 학습 이미지의 겹침 검사를 호출 지점에서 할 수 있음 (CLAUDE.md 규칙 6) | 파트 3 (재학습 데이터 구성), 백엔드 (inspection_id로 이미지 읽기) |
| C4 | train_data_snapshot 구성 | `train` / `pool` 건수 | `train`, `pool` 건수에 재현용 `seed`를 더하고, 키 추가 허용 범위를 4장에 명시 | 같은 시드, 같은 데이터면 같은 모델이라는 재현성을 기록으로 확인 | metrics.json을 읽는 쪽 (파트 3, 백엔드 `/models/{version}`, 모델 관리 화면). 모르는 키는 무시 |
| C5 | 임계값의 출처 | "모델 버전에 함께 기록"만 있음 | 추론은 설정 파일이 아니라 불러온 모델 버전의 metrics.json `threshold`를 쓴다. 설정 파일(`predictor.toml`)의 threshold는 새 모델에 기록할 기본값이다 | 임계값이 모델 버전에 묶여야 판정 로그의 `threshold`와 재현 결과가 맞음 | 파트 3 (승격·롤백 때 임계값이 모델과 함께 이동), 백엔드 (`/predict`가 `predictor.threshold`를 Prediction에 기록) |
| C6 | 승격 게이트 근거 문장 | 스파이크 수치(옛 부품 그림 기준) | 현재 수집기·드리프트 강도 기준 재측정값으로 교체 (규칙 G1~G3은 바꾸지 않음, 설명 문장만) | 근거가 옛 그림의 수치라 현재 구현과 달랐음 | 설명 문장만 (코드 영향 없음). e2e-scenario.md 3단계 합격 기준의 옛 수치 문구 |

**상태: C3~C6 승인됨 (2026-10-18).** 본 문서의 아래 본문이 그대로 유효합니다.

### v1.1 이후 변경 제안 (v1.2)

| # | 변경 | v1.1 | 제안 | 이유 | 영향 범위 |
|---|---|---|---|---|---|
| C7 | 파트 3 보조 테이블 보강 | training_run: run_id, status, trigger, started_at, finished_at, result_version | 컬럼 3개 추가(step, error_message, gate_result), status 영문 저장, 승인 요청 중복 방지 제약, 롤백 시 approved_by 의미 명시, ID 형식·audit action 값 명시 (5~7장) | 재학습 화면의 진행 단계·실패 메시지·게이트 수치(screens.md 4장, e2e 4~5단계)를 담을 곳이 계약에 없었음. 같은 후보에 승인 요청이 둘 생겨 두 번 처리되는 사고를 DB에서 막기 위해 | 파트 3 (training_run, approval_request, model_version, audit_log), 백엔드 /training, /approvals, /models. tests/contract(파트 1·2)와 common/schemas.py는 영향 없음 |

**상태: C7 승인됨 (2026-10-25).** 본 문서의 아래 본문이 그대로 유효합니다.

### v1.2 이후 변경 제안 (v1.3)

| # | 변경 | v1.2 | 제안 | 이유 | 영향 범위 |
|---|---|---|---|---|---|
| C8 | training_run 시각 필드의 비는 시점, pool 편입 규칙 명시 (2장, 7장) | started_at, finished_at이 언제 비는지 적혀 있지 않고, 편입은 "재학습 시 train으로 편입"만 있음 | started_at은 running이 될 때, finished_at은 succeeded/failed가 될 때 기록하며 그 전에는 비어 있음. pool 편입은 재학습의 data 단계에서 읽은 행만 대상으로 하고, 편입과 후보 등록(모델 버전 등록, 게이트 결과 기록 포함)은 한 번에 일어나며 하나라도 실패하면 모두 일어나지 않음. 게이트 미통과(succeeded)여도 편입하고, 학습이 failed면 편입하지 않음. 학습 중에 새로 생긴 pool 행은 pool로 남음 | 재학습 화면이 시각이 비어 있을 수 있음을 전제로 표시해야 하고, 데이터 구성 규칙의 모호함을 없애기 위해 | 파트 3, 백엔드 /training, /inspections(data_split). tests/contract(파트 1·2)와 common/schemas.py는 영향 없음 |

**상태: C8 승인됨 (2026-10-25).** 본 문서의 아래 본문이 그대로 유효합니다.

### v1.3 이후 변경 제안 (v1.4)

| # | 변경 | v1.3 | 제안 | 이유 | 영향 범위 |
|---|---|---|---|---|---|
| C9 | audit_log action 값 추가 (7장) | `gate_pass` / `gate_fail` / `approve` / `reject` / `rollback`. 새 모델 로딩 실패(e2e X2)의 기록 이름은 아직 없음 | `approve_fail`, `rollback_fail` 추가. 승인·롤백 때 새 모델을 불러오지 못해 처리하지 못한 경우의 기록이며, 이때 stage·승인 요청·배포 이력은 바뀌지 않음 | 로딩 실패(X2)를 감사 로그에 남겨야 하는데 감사 로그에는 상세 칸이 없어서, 어떤 작업이 실패했는지 이름으로 구분해야 함 | 파트 3, 백엔드 /audit-logs. DB 스키마 변경 없음. tests/contract(파트 1·2)와 common/schemas.py는 영향 없음 |

**상태: C9 승인됨 (2026-10-25).** 본 문서의 아래 본문이 그대로 유효합니다.

---

## 1. 공통 규칙

| 항목 | 규칙 |
|---|---|
| 이미지 | 128×128, 흑백(1채널), PNG, 픽셀 0~255 (uint8) |
| 라벨 | `OK`(정상) / `NG`(부품 누락). 모델이 예측하는 양성(positive) 클래스는 **NG** |
| 시각 | ISO 8601 문자열 (예: `2026-10-17T21:30:00+09:00`) |
| ID | 문자열. 검사 `INS-000001`, 판정 `PRD-000001` 형식 권장 |
| 전처리 | `common/preprocess.py` 한 곳에서만 수행. 학습·추론이 같은 함수를 사용 |
| 점수(score) | NG일 확률, 0.0~1.0 |
| 기본 임계값 | 0.5 (score ≥ 임계값이면 NG). 변경 가능하며 모델 버전(metrics.json)에 함께 기록. **추론은 설정 파일이 아니라 불러온 모델 버전에 기록된 값을 쓴다.** 설정 파일(`predictor.toml`)의 threshold는 새로 학습하는 모델에 기록할 기본값이다 (C5) |

---

## 2. inspection (검사 1건) — 파트1 → 파트2·3

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| inspection_id | string | ✅ | 고유 ID |
| captured_at | datetime | ✅ | 촬영(생성) 시각 |
| image_path | string | ✅ | 이미지 파일 경로 (프로젝트 루트 기준 상대경로) |
| true_label | enum(OK/NG) | ✅ | 가상 정답 |
| missing_parts | list[int] | ✅ | 누락된 부품 번호 (정답, 정상이면 빈 목록). 1차 범위에서는 모델이 쓰지 않으나 시뮬레이터는 기록해 둔다 |
| scenario | string | ✅ | `normal` 또는 `drift` |
| drift_type | string | ⬜ | scenario가 drift일 때: `dark` / `noise` / `shift` 등 |
| data_split | enum | ✅ | `train` / `pool` / `test_normal` / `test_drift` |

**검증 규칙 (파트1이 저장 전에 확인)**

| 규칙 | 실패 시 |
|---|---|
| 이미지가 128×128, 1채널인가 | 저장하지 않고 오류 기록 |
| inspection_id가 중복되지 않는가 | 저장하지 않고 오류 기록 |
| 파일이 열리는가 (손상 여부) | 저장하지 않고 오류 기록 |
| true_label이 OK이면 missing_parts가 비어 있는가 | 저장하지 않고 오류 기록 |

**data_split 사용 규칙**

| 값 | 용도 | 변경 가능 여부 |
|---|---|---|
| train | v1 및 재학습의 학습 데이터 | 한 번 정해지면 유지 |
| pool | 운영 중 계속 쌓이는 신규 데이터, 재학습 때 train에 합류 | 재학습 시 train으로 편입 (C8: 재학습의 data 단계에서 읽은 행만 대상. 편입과 후보 등록(모델 버전 등록, 게이트 결과 기록 포함)은 한 번에 일어나며 하나라도 실패하면 모두 일어나지 않는다. 게이트 미통과여도 편입하고, 학습이 failed면 편입하지 않는다. 그 뒤 새로 생긴 pool 행은 pool로 남는다) |
| test_normal | 고정 테스트셋 (정상 환경) 500건 (정상 400 + 불량 100) | **고정, 학습에 절대 사용 금지** |
| test_drift | 고정 테스트셋 (드리프트 환경) 500건 (정상 400 + 불량 100) | **고정, 학습에 절대 사용 금지** |

---

## 3. prediction (판정 1건) — 파트2 → 파트3·프론트

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| prediction_id | string | ✅ | 고유 ID |
| inspection_id | string | ✅ | 검사 FK |
| model_version | string | ✅ | 사용한 모델 버전 (예: `v1`) |
| predicted_label | enum(OK/NG) | ✅ | 판정 결과 |
| score | float | ✅ | NG 확률 (0.0~1.0) |
| threshold | float | ✅ | 판정에 사용한 임계값 (= 사용한 모델 버전의 metrics.json `threshold`) |
| latency_ms | int | ✅ | 추론 소요 시간 (5초 = 5000ms 이내가 요건) |
| predicted_at | datetime | ✅ | 판정 시각 |

**파생 값 (저장하지 않고 조회 시 계산)**: 정오 여부 = `predicted_label == inspection.true_label`

---

## 4. 모델 추론 계약 (파트2의 입출력)

| 구분 | 내용 |
|---|---|
| 입력 | 이미지 1장 (128×128 흑백) |
| 출력 | `{predicted_label, score, model_version, latency_ms}` |
| 모델 저장 형식 | `models/<version>/model.pt` (가중치) + `models/<version>/metrics.json` (지표) |
| 모델 로딩 | 추론부는 production 버전 파일(model.pt, metrics.json)을 읽어 사용. 임계값도 같은 버전의 metrics.json에서 읽는다 (C5). 로딩 실패 시 기존 모델 유지 |
| 학습 진입점 | 학습 이미지 배열(N, 128, 128, uint8)과 라벨 목록을 받아 새 버전을 만들고 metrics.json을 남김. inspection_id → 이미지 변환은 파트 3 또는 백엔드가 맡는다. 평가 데이터·버전 이름 등 상세는 `parts/predictor/README.md` 참조 (C3) |
| 평가 진입점 | 모델과 평가 이미지 배열·라벨·임계값을 받아 지표를 계산 (C3) |

**metrics.json 형식**

| 필드 | 설명 |
|---|---|
| version | 모델 버전 |
| threshold | 사용한 임계값 |
| normal | test_normal 결과: accuracy, recall, precision, f1, confusion_matrix([[TN, FP],[FN, TP]]), n |
| drift | test_drift 결과: 위와 동일 |
| train_data_snapshot | 학습에 사용한 이미지 개수와 구성: train·pool 건수와 재현용 seed (키 규칙은 아래, C4) |
| created_at | 생성 시각 |

**train_data_snapshot 키 규칙 (C4):** `train`, `pool`은 건수(0 이상 정수)이고 둘의 합이 학습 이미지 수와 같다. 재현에 도움이 되는 정수 항목(예: `seed`)은 키를 추가해도 된다. 읽는 쪽은 모르는 키를 무시하고 오류를 내지 않는다. 기존 키의 삭제·이름 변경·의미 변경은 변경 제안 대상이다.

---

## 5. model_version (모델 버전) — 파트3

| 필드 | 타입 | 설명 |
|---|---|---|
| version | string | `v1`, `v2` … |
| stage | enum | `candidate` / `staging` / `production` / `archived` / `rejected` |
| metrics | json | metrics.json 내용 |
| train_data_snapshot | json | 학습 데이터 구성 요약 (4장 키 규칙과 같음) |
| created_at | datetime | 생성 시각 |
| approved_by | string | 승인자 (승인 전에는 비어 있음) |
| approved_at | datetime | 승인 시각 |

**approved_by / approved_at의 의미 (C7):** "마지막으로 production에 올린 사람과 시각"이다. 승인(staging → production)과 롤백(archived → production) 때 각각 처리한 관리자와 시각으로 갱신한다. 처음 승인 기록은 approval_request에, 승격·롤백 이력은 deployment에 남으며, 롤백해도 approval_request 원본은 바뀌지 않는다. `sync`로 처음 등록한 v1처럼 사람이 올리지 않은 production은 비어 있다.

**상태 전이 (기획서 7장)**

| 현재 상태 | 조건 | 다음 상태 |
|---|---|---|
| candidate | 승격 게이트 통과 | staging (승인 대기) |
| candidate | 승격 게이트 미통과 | rejected (사유: 게이트 미통과) |
| staging | 관리자 승인 | production (기존 production은 archived) |
| staging | 관리자 거절 | rejected |
| archived | 관리자 롤백 (사유 필수) | production (현재 production은 archived) |

**승격 게이트 (v1.0 확정)**

| # | 평가셋 | 지표 | 조건 | 의미 |
|---|---|---|---|---|
| G1 | test_drift | 정확도(accuracy) | 후보 ≥ 현재 production **+1%p** | 드리프트 환경에서 실제로 나아졌는가 |
| G2 | test_drift | 재현율(recall) | 후보 ≥ 현재 production **−1%p** | 불량을 더 놓치지 않는가 |
| G3 | test_normal | 정확도(accuracy) | 후보 ≥ 현재 production **−1%p** | 정상 환경 성능이 깎이지 않았는가 |

- G1~G3을 **모두** 만족해야 통과(staging)한다. 하나라도 미통과면 rejected이며 사유에 미통과 조건을 적는다.
- 현재 production이 없는 첫 모델(v1)에는 게이트를 적용하지 않는다.
- 각 테스트셋은 500건(정상 400건 + 불량 100건)이므로 G2의 재현율 1%p는 **불량 100건 중 1건 차이**에 해당한다.
- G1·G3의 정확도 1%p는 각 테스트셋 500건 중 **5건**에 해당한다.
- 근거 (C6): 3-D 측정. 조건은 train 2,000장 / pool 500장(드리프트 환경) / 테스트셋 각 500장, 드리프트 강도 dark 0.18 · noise 0.24 · shift 0.11. 시드 42 기준 v1→v2 test_drift 정확도 0.854→0.970 (3시드 평균 0.809→0.933), 재현율 1.000→1.000, test_normal 정확도 1.000→1.000으로 G1~G3 통과. 3시드(42, 43, 44) 모두 G1~G3을 통과했다. 재현: `python -m scripts.make_dataset`, `python -m scripts.measure_baseline --seeds 42,43,44` (결과 파일 `data/baseline_report.txt`는 저장소에 포함되지 않으므로 재실행으로 만든다). 테스트셋 규모 변경은 별도 승인이 필요하다.

---

## 6. approval_request (승인 요청) — 파트3

| 필드 | 타입 | 설명 |
|---|---|---|
| approval_id | string | 고유 ID |
| model_version | string | 승인 대상 버전 |
| requested_by | string | 요청자 (시스템 자동 생성이면 `system`) |
| status | enum | `pending` / `approved` / `rejected` |
| reason | string | 승인/거절 사유 (거절 시 필수) |
| handled_by | string | 처리자 |
| handled_at | datetime | 처리 시각 |
| created_at | datetime | 요청 생성 시각 |

**제약 (C7):** 같은 model_version에 status가 `pending`인 승인 요청은 동시에 1개만 존재한다. 이미 있는데 새로 만들려 하면 만들지 않고, API는 409(conflict)로 응답한다.

---

## 7. 보조 테이블

| 테이블 | 주요 필드 | 용도 |
|---|---|---|
| training_run | run_id, status, trigger, started_at, finished_at, result_version, step, error_message, gate_result | 재학습 실행 이력 |
| deployment | deployment_id, version, action(deploy/rollback), reason, actor, at | 배포·롤백 이력 |
| audit_log | log_id, actor, role, action, target, at | 승인·롤백 등 감사 기록 |

**ID 형식 (C7):** approval_id=`APR-000001`, run_id=`RUN-000001`, deployment_id=`DEP-000001`, log_id=`LOG-000001` (접두어 + 6자리 숫자).

**training_run 값 규칙 (C7)**

- status: `queued`(대기) / `running`(실행) / `succeeded`(성공) / `failed`(실패). 저장과 API 응답은 영문, 괄호 안 한글은 화면 표시 이름이다.
- trigger: `manual` / `count` / `performance`.
- status와 게이트의 구분: **학습이 끝까지 돌았으면 status는 `succeeded`**이고, 게이트 통과 여부는 `gate_result.passed`가 담당한다. 게이트에서 떨어진 모델도 `result_version`을 갖고 model_version의 stage는 `rejected`이다. `failed`는 **학습 도중 오류가 났을 때만** 쓰며, 이때는 모델 버전이 만들어지지 않아 `result_version`이 비어 있다.
- step: 진행 단계. `data`(데이터 구성) / `train`(학습) / `evaluate`(평가) / `gate`(게이트). 대기 중이면 비어 있다.
- error_message: status가 `failed`일 때의 오류 메시지. 아니면 비어 있다.
- result_version: 학습이 끝까지 돌아 만들어진 모델 버전. 아니면 비어 있다.
- started_at (C8): status가 `running`이 될 때 기록한다. `queued`에서는 비어 있다.
- finished_at (C8): status가 `succeeded` 또는 `failed`가 될 때 기록한다. 그 전에는 비어 있다.
- gate_result: 게이트 결과 JSON. 게이트 전에는 비어 있다. 키는 아래와 같다.
  - `baseline_version`: 비교 기준이 된 운영 버전 (첫 모델이면 null)
  - `passed`: 전체 통과 여부
  - `checks`: G1~G3 각각 `{id, passed, candidate(후보 값), baseline(기준 값), required(요구 조건), message(근거 문장)}`의 목록
  - 게이트에서 떨어진 모델의 미통과 사유는 여기에 남는다.
  - **첫 모델(비교 대상인 production이 없음)**: `{"baseline_version": null, "passed": true, "checks": []}`. 게이트를 적용하지 않았다는 뜻이며, 화면은 `baseline_version`이 null이면 "게이트 미적용"으로 표시한다.

**audit_log 값 규칙 (C7)**

- action: `gate_pass` / `gate_fail` / `approve` / `reject` / `rollback` / `approve_fail` / `rollback_fail` (C9). target은 대상 모델 버전(예: `v2`), role은 `operator` / `admin` / `system`.
- `approve_fail`, `rollback_fail` (C9): 새 모델 파일을 불러오지 못해 승인·롤백을 하지 않았다는 기록(e2e X2)이다. 이때 stage·승인 요청·배포 이력은 바뀌지 않는다. target은 시도한 버전이고, actor·role은 요청한 관리자다.

---

## 8. 파트 간 계약 요약 (인수인계용)

| 경계 | 보내는 쪽 | 받는 쪽 | 약속 |
|---|---|---|---|
| 데이터 계약 | 파트1 (수집) | 파트2·3 | inspection 스키마 + 이미지 규칙(2장). 규칙 위반 데이터는 저장하지 않는다 |
| 모델·로그 계약 | 파트2 (추론) | 파트3·프론트 | 추론 입출력(4장) + prediction 스키마(3장) + 모델 저장 형식 |

## 9. 계약 테스트 (`tests/contract/`, 구현 시 작성)

| 대상 | 확인 내용 |
|---|---|
| 수집 (`parts/collector`, 시뮬레이터 포함) | 생성된 이미지가 128×128 흑백인가, 필수 필드가 모두 있는가, 정상 라벨이면 missing_parts가 비어 있는가 |
| 추론 (`parts/predictor`) | 이미지 1장을 넣으면 `{predicted_label, score, model_version, latency_ms}`가 나오는가, score가 0~1인가, 5초 이내인가 |
| 모델 저장 | `models/<version>/`에 model.pt와 metrics.json이 있고 형식이 맞는가 |

개선하신 분은 코드를 바꾼 뒤 `pytest tests/contract`를 실행해 통과하면, 다른 파트가 깨지지 않았음을 확인할 수 있습니다.
