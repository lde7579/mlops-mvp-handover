# 조원 인수 가이드 (handover)

> 이 문서만 읽으면 시작할 수 있게 쓴 가이드입니다. 작성 시점의 계약은 **v1.4 (동결, 승인일 2026-10-25)** 입니다.
> 압축본에는 `.git`이 없으므로 계약 버전은 **`docs/data-contract.md`의 첫 줄(제목)과 "변경 이력"** 으로 확인합니다.
> 2026-10-31 갱신: 성능 저하 감지 기준 N=30, T=0.95 확정(api-v1.md A27, 계약 v1.4는 변경 없음), 감지 측정 스크립트(`scripts/measure_detection.py`) 추가.
> 2026-11-01 갱신: 시연 초기화 도구(`scripts/reset_demo.py`)와 시연 체크리스트(`docs/demo-checklist.md`) 추가 (프로젝트 담당자 전담 도구, 조원은 실행·수정 불필요). 압축 파일 수와 시험 개수는 달라질 수 있으니 개수가 아니라 "실패 0, skipped 0"으로 판단합니다.

## 1. 압축에 들어 있는 것과 없는 것

- **들어 있음:** `docs/`, `common/`, `parts/`, `backend/`, `services/`, `frontend/`, `tests/`, `scripts/`, `CLAUDE.md`, `pyproject.toml`, `requirements.txt`, `.gitignore`, `.gitattributes`
- **없음:** `.git`, `.venv`, `data/`, `models/`, `__pycache__`, `.pytest_cache`, `spike/`
  - `.venv`, `data/`, `models/`는 아래 2·3장대로 직접 만듭니다 (용량이 크고 컴퓨터마다 다시 만드는 것이 안전합니다). `models/`는 압축에 없으므로 PC마다 3장의 두 명령으로 만듭니다.
  - 문서에 나오는 `spike/`는 과거 실험 기록이고 압축에는 포함하지 않는다.
- 압축은 `python -m scripts.make_handover_zip`으로 만듭니다 (허용 목록의 파일만 담아 `.venv` 같은 실수를 막음).
- **보내는 쪽 체크:** 압축 전 `git status`가 비어 있는지 확인한다.

## 2. 환경 만들기 (Windows PowerShell, 압축을 푼 `mlops-mvp` 폴더에서)

Python **3.11 이상**과 인터넷이 필요합니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

- **폴더 경로는 짧게** 합니다 (예: `C:\work\mlops-mvp`). 압축을 깊은 폴더에 풀면 Windows 경로 길이 제한(260자) 때문에 `pip install`이 `No such file or directory` 오류로 실패할 수 있습니다.
- `py` 명령이 없는 PC에서는 `py -3.11 -m venv .venv` 대신 `python -m venv .venv`를 쓰고, 먼저 `python --version`이 3.11 이상인지 확인합니다.
- 활성화에서 "스크립트를 실행할 수 없습니다" 오류가 나면 활성화를 건너뛰고 `.\.venv\Scripts\python.exe -m pytest tests/ -rs`처럼 파이썬 경로를 직접 쓰면 됩니다.
- 앞으로의 명령은 모두 같은 폴더(`mlops-mvp`)에서 실행합니다.
- `requirements.txt`의 주석은 영어로 씁니다 (한국어 Windows의 pip가 한글 주석을 cp949로 읽다가 오류를 내는 것을 막기 위해서입니다). 그래도 설치가 인코딩 오류로 실패하면 `$env:PYTHONUTF8=1`을 먼저 실행한 뒤 `pip install -r requirements.txt`를 다시 합니다.
- 하위 패키지(직접 적지 않았지만 함께 설치되는 패키지)는 새 venv에서 **30개**가 함께 설치되고, 설치(torch 포함)에 약 **4분** 걸립니다 (2026-10-25 새 폴더·새 venv 실측: torch 146초 + 나머지 100초).
- 테스트 중 폴더 링크(정션)를 만듭니다. 만들 권한이 없는 PC에서는 `tests/backend` 일부가 **실패**합니다 (건너뛰지 않고 실패하는 것이 정상 동작입니다).
- 백엔드는 `uvicorn` 프로세스 **1개만** 지원합니다 (`--workers 2` 이상이나 서버 여러 개 동시 실행은 안 됩니다. 검사 번호 잠금이 프로세스 안에서만 작동하기 때문입니다).

## 3. 데이터 만들기 (필요할 때만)

```powershell
python -m scripts.make_dataset
```

- 약 80초, 3,500장(train 2,000 / pool 500 / test_normal 500 / test_drift 500)을 `data/dataset/`에 만듭니다. 같은 명령이면 같은 이미지가 나옵니다.
- 이미 있으면 아무것도 바꾸지 않고 안내만 합니다. 다시 만들려면 `--regenerate` (`data/dataset`만 지웁니다).
- **`pytest`만 돌릴 때는 필요 없습니다** (테스트는 임시 폴더에 작은 데이터를 스스로 만듭니다).

### 모델 만들기 (`models/`, 서버를 실제로 돌릴 때만)

```powershell
python -m scripts.train_v1
python -m backend.registry sync --production v1
```

- `models/`는 압축에 없으므로 **PC마다** 이 두 명령으로 만듭니다 (`data/dataset`이 먼저 있어야 합니다). `pytest`만 돌릴 때는 필요 없습니다.
- 모델이나 전처리를 바꾼 뒤에는 `python -m scripts.measure_latency`로 판정 시간 5초 요건을 다시 확인합니다 (약 3~4분, `models/v1`·`backend/`·torch 필요. 실제 `data/`·`models/`는 건드리지 않고, 기존 `data/latency_report.txt`는 `--overwrite` 없이는 덮어쓰지 않습니다).
- 감지 기준(N·T)은 `python -m scripts.measure_detection`으로 측정했습니다. 프로젝트 담당자용이라 조원은 실행하지 않아도 됩니다. `data/dataset`, `models/v1`, torch가 필요하고 약 10분 걸리며, `data/detection_report.txt`를 덮어씁니다 (서버·DB는 쓰지 않고 `models/v1`은 읽기만 합니다). `pytest`는 이 스크립트의 합성 데이터 시험만 돌립니다.

## 4. 합격 기준

```powershell
pytest tests/ -rs
```

- **실패 0, skipped 0** 이 합격입니다. 건수는 테스트가 추가되면 달라질 수 있으니 **건수가 아니라 "실패 0, skipped 0"** 로 판단합니다.
- 계약 테스트만 보려면 `pytest tests/contract -rs` (작성 시점 5개). 이것이 인수 기준입니다.

## 5. 수정해도 되는 곳 / 안 되는 곳

- **수정 가능:** `parts/collector/`, `parts/predictor/` (코드, 설정 파일 값, README). 입출력 형식만 계약을 지키면 내부 구현은 자유롭게 바꿀 수 있습니다.
- **`tests/`의 보조 테스트**(`tests/test_collector.py`, `tests/test_predictor.py` 등)는 담당 파트 구현에 맞춰 수정·추가할 수 있습니다. `pytest`는 `tests/`만 돌립니다.
- **수정 금지:** `common/`, `scripts/`, `docs/`, `tests/contract/` (계약과 인수 기준). 고쳐야 하면 6장 절차로 제안합니다.
- **`backend/`, `services/`는 프로젝트 담당자(본인) 담당**이므로 조원은 수정하지 않습니다 (이 코드와 `tests/backend/`, `tests/training/`는 수정 대상이 아닙니다).
- **`scripts/reset_demo.py`(시연 초기화)와 `docs/demo-checklist.md`(시연 체크리스트)도 프로젝트 담당자(본인) 전담 도구**입니다. 조원은 실행하지 않아도 되고 수정할 필요도 없습니다 (조원이 `scripts/`를 고치면 `tests/test_reset_demo.py`만 영향을 받습니다).
- **`frontend/`(Streamlit 화면)도 프로젝트 담당자(본인) 전담**이므로 조원은 수정하지 않습니다 (폴더가 있고, 이 코드와 `tests/frontend/`는 수정 대상이 아닙니다).
- **코드를 바꾸면 먼저 깨지는 시험:** 아래 이름을 바꾸면 해당 시험이 실패해서 알려 줍니다. 깨지면 이름과 동작을 유지하거나 담당자에게 알립니다. 합격 기준은 `tests/contract` 통과(실패 0, skipped 0)입니다.
  - `Collector._next_id` (백엔드 자동 생성이 번호를 이어 가는 데 사용, `tests/backend/test_simulator_generate.py`)
  - `Collector._save_png` (`tests/test_collector.py`의 통계 시험과 `tests/conftest.py`의 `tiny_data`가 PNG 저장을 생략하는 데 사용. 이름이 바뀌면 `tiny_data`를 쓰는 시험이 전부 먼저 실패합니다)
  - `parts.predictor.config.load_config`와 `models_dir` 설정 (`backend/registry.py`가 Predictor와 같은 `models/` 폴더를 보는 데 사용)
  - `scripts/measure_latency.py`의 `TIMER_TARGETS` (제품 함수 이름이 바뀌면 `tests/test_measure_latency.py`가 먼저 알려 줍니다)
  - `Collector.next_sample(label=, drift_type=)` 공개 옵션 (백엔드 자동 생성이 사용합니다. 이 옵션 이름과 의미를 바꾸면 안 됩니다)
  - `parts.collector.simulator.make_image(rng, unfastened, drift_type, strength)`와 `DRIFT_TYPES` (`scripts/sweep_drift.py`, `scripts/measure_detection.py`가 새 이미지를 만드는 데 사용합니다. 바꾸면 `tests/test_sweep_drift.py`, `tests/test_measure_detection.py`가 먼저 알려 줍니다)
- 설정 값 주의
  - `image_size = 128`, `part_count = 6`은 계약 값이라 바꾸면 오류가 납니다.
  - 드리프트 강도(`dark 0.18`, `noise 0.24`, `shift 0.11`)를 바꾸면 데이터셋을 다시 만들어야 하므로 먼저 협의합니다.
- `spike/` 코드는 import하지 않습니다 (CLAUDE.md 규칙).

## 6. 계약 변경이 필요한 경우

스키마 필드 추가, 허용값·단위·파일 형식·입출력 형식 변경이 필요하면 **코드를 먼저 바꾸지 않습니다.**

1. `docs/data-contract.md` 0장에 변경 제안 행을 올립니다 (C7부터: 변경 / 현재 / 제안 / 이유 / 영향 범위).
2. 승인을 받은 뒤에 문서와 코드를 함께 바꾸고 계약 버전을 올립니다.

## 7. 돌려받을 때 (받는 쪽 절차)

1. 받은 압축을 **새 폴더**에 풀고, `docs/data-contract.md` 첫 줄과 변경 이력에서 계약 버전을 확인합니다.
2. 원본과 비교해 `docs/`, `common/`, `scripts/`, `tests/contract/`가 바뀌지 않았는지 봅니다. 예: `git diff --no-index <원본 폴더> <받은 폴더>`
3. 2장대로 환경을 만들고 `pytest tests/ -rs`로 **실패 0, skipped 0** 을 확인합니다.
4. `pytest tests/contract -rs`로 계약 테스트 통과를 확인합니다.
5. 바뀐 파일 목록이 `parts/`와 `tests/` 안인지 검토합니다. 문서가 바뀌었다면 변경 제안 절차(6장)를 거쳤는지 확인합니다.
