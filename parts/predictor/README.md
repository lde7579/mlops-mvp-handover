# parts/predictor — 파트 2: 모델 추론

- 담당: 조원
- 입출력 약속: `docs/data-contract.md` (v1.1)의 3·4장 prediction 스키마와 모델 추론 계약
- 인수 기준: `pytest tests/contract`의 추론·모델 저장 항목 통과
- 전처리는 `common/preprocess.py`만 사용한다 (학습·추론 공유)
- `spike/` 코드는 import하지 않는다. 필요한 부분은 복사해서 계약 테스트와 함께 새로 넣는다

## 사용법

```python
from parts.predictor import Predictor, train, evaluate

# 학습: models/v1/model.pt, metrics.json 을 만든다 (평가 데이터는 normal, drift 둘 다 필수)
metrics = train(images, labels, version="v1",
                eval_normal=(normal_images, normal_labels), eval_drift=(drift_images, drift_labels),
                snapshot={"train": 2000, "pool": 0})   # snapshot 은 생략 가능

# 추론: 모델 버전을 올린 뒤 이미지 1장씩 판정한다
p = Predictor()                  # 설정: parts/predictor/predictor.toml
p.load("v1")                     # 실패하면 ModelLoadError, 기존에 올려 둔 모델은 그대로 유지
out = p.predict(image)           # image: uint8 (128, 128)
out.predicted_label, out.score, out.model_version, out.latency_ms
p.version, p.threshold           # Prediction 한 건을 만들 때 쓰는 값
```

- `images`는 uint8 `(N, 128, 128)` 배열, `labels`는 `"OK"`/`"NG"` 목록이다.
- `evaluate(model, images, labels, threshold)`: 모델(`SmallCNN` 또는 `Predictor`), 평가 이미지 배열(uint8 `(N, 128, 128)`), 라벨 목록(`"OK"`/`"NG"`), 임계값을 받아 accuracy, recall, precision, f1, 혼동행렬(`[[TN, FP], [FN, TP]]`), n을 계산한다.
- NG 판정은 `score >= threshold` (계약 1장). 임계값은 `predictor.toml`이 아니라 **불러온 모델 버전의 `metrics.json`에 기록된 값**을 쓴다.
  `predictor.toml`의 `threshold`는 새로 학습하는 모델에 기록할 기본값이다.
- `latency_ms`는 이미지 검사 + 전처리 + 추론에 걸린 시간(정수 ms)이다.
- 버전 이름은 `v1`, `v2`처럼 `v` 다음에 숫자만 허용하고, 이미 있는 버전은 덮어쓰지 않는다. 저장은 임시 폴더에 다 쓴 뒤 옮겨서 반쪽짜리 폴더가 남지 않는다.
- 평가 이미지가 학습 이미지와 하나라도 겹치면 학습을 거부한다 (테스트셋은 학습에 쓰지 않는다).
- 같은 시드(기본 42), 같은 데이터면 같은 모델이 나온다 (학습 동안만 결정적 연산을 켠다).
- `models/`는 `.gitignore`에 있어 커밋되지 않는다.

## 계약 v1.1 과의 관계

처음 구현할 때 계약 v1.0과 달랐던 부분은 변경 제안 C3~C5로 계약 v1.1에 반영되었다 (`docs/data-contract.md` 0장 두 번째 표).

- 학습 진입점 (C3): `train()`은 inspection_id 목록이 아니라 **이미지 배열과 라벨 목록**을 받는다. 평가용 이미지·라벨(normal, drift)과 버전 이름도 함께 받는다.
  inspection_id를 이미지로 바꾸는 부분(파일·DB 읽기)은 파트 3이나 백엔드가 맡는다.
- `train_data_snapshot` (C4): `{"train": 건수, "pool": 건수, "seed": 시드}`로 기록한다. `seed`는 재현을 위해 더한 키이고, 키 추가 허용 범위는 계약 4장에 있다.
- 임계값 (C5): 추론은 `predictor.toml`이 아니라 불러온 모델 버전의 `metrics.json` 값을 쓴다. `predictor.toml`의 `threshold`는 새 모델에 기록할 기본값이다.
- 버전 이름은 호출하는 쪽이 정한다 (계약에 부여 규칙은 없다). 자동으로 다음 번호를 정하는 일은 파트 3의 모델 레지스트리 몫이다.
