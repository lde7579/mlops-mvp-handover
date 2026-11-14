# backend — FastAPI 백엔드 (4-B: 검사, 추론, 판정·모델 조회, 재학습)

- 담당: 본인 (조원 수정 금지)
- 기준 문서: `docs/api-v1.md`. 지금 있는 엔드포인트는 아래 표와 같고, 나머지(승인·배포·모니터링 등)는 이후 단위에서 만든다.

| 메서드 | 경로 | 권한 | 설명 |
|---|---|---|---|
| GET | `/health` | 공개 | 백엔드·DB·모델 상태. `model.loaded`는 운영 모델을 실제로 불러와 두었는가 |
| POST | `/auth/login` | 공개 | 토큰 발급 |
| POST | `/simulator/generate` | 운영자 이상 | 검사 1건 생성 (201). 본문 `{"scenario": "normal" 또는 "drift"}`, 없으면 normal |
| GET | `/inspections` | 로그인 | 목록 `{"items": [...], "total": N}`. 필터 `scenario`, `data_split`, `since`, `until`, `limit`(1~1000, 기본 50), `offset` |
| GET | `/inspections/{id}` | 로그인 | 상세 (판정 목록 `predictions` 포함) |
| GET | `/inspections/{id}/image` | 로그인 | PNG 이미지 |
| POST | `/predict` | 운영자 이상 | 검사 1건 판정·저장 (201). 본문 `{"inspection_id": "INS-000001"}`. 같은 검사를 여러 번 판정해도 된다 |
| GET | `/predictions` | 로그인 | 판정 이력 `{"items": [...], "total": N}`, 최신 먼저. 항목은 판정 8개 필드 + `true_label`(정답)·`correct`(정오). 필터 `model_version`, `inspection_id`, `wrong`(소문자 `true`=오판만, `false`=정판만), `limit`, `offset` |
| GET | `/predictions/stats` | 로그인 | 판정 통계 `{"total", "ng_count", "ng_rate"(판정 기준), "recent": {"n", "accuracy", "recall", "confusion_matrix"}}`. 쿼리 `recent`(최근 N건, 1~1000, 기본 100), `model_version` |
| GET | `/models` | 로그인 | 모델 버전 목록 `{"items": [...], "total": N}`, 최신 먼저, `stage` 필터 |
| GET | `/models/production` | 로그인 | 현재 운영 모델 (없으면 404) |
| GET | `/models/{version}` | 로그인 | 모델 버전 상세 (`v숫자` 형식이 아니거나 없는 버전은 404) |
| POST | `/training/runs` | 운영자 이상 | 재학습 시작 (201, `queued` 한 건). 본문 `{"trigger": "manual"}`(생략 가능, manual 만 허용). 실행 중이거나 신규 pool 이 0건이면 409, 데이터셋이 없으면 503 |
| GET | `/training/runs` | 로그인 | 실행 이력 `{"items": [...], "total": N}`, 최신 먼저. 필터 `status`, `limit`, `offset` |
| GET | `/training/runs/{id}` | 로그인 | 실행 상세 (`RUN-숫자` 형식이 아니거나 없으면 404). 항목은 계약 7장 9개 필드 + `train_data_snapshot` |

- **단일 워커만 지원한다.** `uvicorn`을 프로세스 하나로 실행해야 한다 (`--workers 2` 이상, 또는 서버를 여러 개 동시에 실행하는 것은 지원하지 않는다). 검사 번호(ID)를 겹치지 않게 발급하는 잠금이 프로세스 안에서만 작동하기 때문이다.
- 요청 본문의 **모르는 필드는 400**이다 (오타를 조용히 무시하지 않는다). 시각 필터(`since`, `until`)는 시간대가 있는 ISO 8601이어야 하고, URL의 `+`는 `%2B`로 보낸다.
- 생성한 이미지는 `data/pool/INS-xxxxxx.png`에 저장된다 (`data/`는 git과 조원 전달 압축에서 제외). 이미지 조회는 `data/` 아래의 `.png` 파일만 내보내고, `..`·절대경로·심볼릭 링크·정션은 모두 404다.
- 수집기 설정은 `backend.toml`의 `collector_config`(기본 `parts/collector/collector.toml`)다. 수집기는 첫 생성 요청 때 만들어진다.

## 실행

```powershell
python -m uvicorn backend.main:create_app --factory --port 8000
```

- 브라우저에서 `http://127.0.0.1:8000/docs` 로 API 문서(자동 생성)를 볼 수 있다.
- DB 파일은 `data/mlops.db` 다 (`data/`는 git과 조원 전달 압축에서 제외).

## 재학습 (4-B)

- `POST /training/runs`는 `queued` 기록을 만들고 곧바로 응답한다. 학습은 백그라운드 스레드가 이어서 한다 (`backend/retrain.py`). 단계는 `data` -> `train` -> `evaluate` -> `gate` 이고, 진행 상태는 `GET /training/runs/{id}`를 3~5초마다 불러 확인한다.
- **학습 데이터**: 고정 데이터셋 `data/dataset`의 train + DB 의 `data_split=train` 검사 + DB 의 `data_split=pool` 검사 (이미지는 `data/` 아래 PNG). 평가는 `data/dataset`의 test_normal, test_drift 이고 학습에는 쓰지 않는다. `data/dataset/pool`은 오프라인 측정용이라 읽지 않는다.
- **편입**: 사용한 pool 은 후보 등록과 함께 train 으로 바뀐다. 편입 대상은 data 단계에서 읽은 검사 ID 뿐이라 학습 중에 새로 생긴 pool 은 pool 로 남는다. 게이트 미통과(`succeeded`)여도 편입하고, 학습이 `failed`면 편입하지 않는다 (계약 C8).
- **결과**: 학습이 끝까지 돌면 `succeeded` 이다. 새 버전(`v2`, `v3`...)이 candidate 로 등록되고 게이트(G1~G3)를 거쳐 통과하면 staging + 승인 요청(pending), 미통과면 rejected 이다. 학습 도중 오류만 `failed`(`error_message`)다.
- **동시 실행 금지**: 실행 중(`queued`, `running`)인 학습이 있으면 새 요청은 409. 서버가 학습 중에 꺼졌으면 다음에 켜질 때 남은 기록을 `failed`("서버가 재시작되어 중단되었습니다")로 정리한다. 만들다 만 `models/<버전>/` 폴더는 지우지 않는다 (필요하면 직접 지우거나 `python -m backend.registry sync` 로 candidate 등록).
- 학습 중에도 `/predict`는 막히지 않는다 (추론과 락을 공유하지 않는다). 다만 같은 CPU 를 쓰므로 판정이 느려질 수 있다.
- 시험은 실제 학습 대신 가짜 학습기를 넣는다 (`RetrainService(settings, session_factory, trainer=..., fixed_loader=...)`). 실제 `train()` 호출 방식은 `tests/backend/test_retrain_service.py`의 시험 하나가 지킨다.

## 모델 준비와 판정 (3-D-1)

처음 한 번, 순서대로 실행한다 (모두 프로젝트 루트에서, `data/dataset`이 있어야 한다: `python -m scripts.make_dataset`).

```powershell
python -m scripts.train_v1                              # models/v1 학습 (시드 42 고정, 이미 있으면 덮어쓰지 않고 멈춤)
python -m backend.registry sync --production v1         # DB 에 v1 을 운영(production) 모델로 등록
```

- `models/`와 `data/`는 git·조원 전달 압축에서 제외된다. 모델은 PC마다 위 명령으로 만든다.
- `sync`는 서버가 켜질 때 자동으로 하지 않는다 (운영 모델을 정하는 것은 사람의 결정). 파일은 있고 DB 에 없는 모델은 candidate 로 등록하고, DB 에는 있는데 파일이 없거나 지표가 다른 모델은 지우거나 덮어쓰지 않고 알려 준다. 같은 명령을 여러 번 실행해도 안전하다.
- 운영 모델은 서버가 켜질 때 미리 불러온다 (실패해도 서버는 뜬다). 켜진 뒤에 등록했거나 운영 모델이 바뀌었으면 첫 `/predict`에서 불러온다. `/health`는 모델 파일을 읽지 않고 지금 상태만 알려 준다.
- 운영 모델이 없거나 불러오지 못하면 `/predict`는 **503**(`service_unavailable`)이다. 요청마다 다시 시도하므로 파일을 고치면 서버를 다시 켜지 않아도 복구된다 (같은 실패는 로그에 한 번만 남는다).
- 임계값은 설정 파일이 아니라 불러온 모델의 `metrics.json` 값이다 (계약 C5).

## 시연 계정 (시연 전용, 실제 환경 사용 금지)

| 역할 | 계정 | 비밀번호 |
|---|---|---|
| 운영자 | `operator` | `operator-demo` |
| 관리자 | `admin` | `admin-demo` |

- **시연용으로 공개된 값이다. 실제 환경에서 사용하면 안 된다.** 비밀번호는 `backend.toml`에 해시로만 저장되어 있다.
- 비밀번호를 바꾸려면 `python -m backend.security` 로 해시를 만들어 `backend.toml`의 `password_hash`를 바꾼다.

```powershell
curl -X POST http://127.0.0.1:8000/auth/login -H "Content-Type: application/json" -d "{\"username\": \"admin\", \"password\": \"admin-demo\"}"
```

## 비밀키 (토큰 서명용)

코드와 저장소 파일에 비밀키는 없다. 우선순위는 다음과 같다.

1. 환경 변수 `MLOPS_SECRET_KEY` (32자 이상)
2. `data/secret.key` 파일 (16진수 64자). 없으면 처음 시작할 때 무작위로 만들어 저장한다.
   - 파일이 깨졌거나 짧으면 서버가 시작되지 않고, 덮어쓰지도 않는다. 파일을 지우고 다시 시작하면 새로 만들어진다(이미 발급된 토큰은 모두 무효).

토큰은 8시간 동안 유효하다 (`backend.toml`의 `token_minutes`).

## 권한과 오류 형식

- 역할은 `operator`(운영자), `admin`(관리자)이고 관리자는 운영자 권한을 모두 가진다.
- 새 엔드포인트는 `backend/deps.py`의 `CurrentUser`, `require_role(Role.OPERATOR)`, `require_role(Role.ADMIN)`, `SessionDep`만 붙이면 된다.
- 오류는 항상 `{"error": "이름", "message": "설명"}` 형식이다. 이름은 `bad_request`(400), `unauthorized`(401), `forbidden`(403), `not_found`(404), `method_not_allowed`(405), `conflict`(409), `server_error`(500), `service_unavailable`(503)다. 입력 검증 실패는 422가 아니라 400이다.

## 테스트

```powershell
pytest tests/backend -q
```
