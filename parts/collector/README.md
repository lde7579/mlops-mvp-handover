# parts/collector — 파트 1: 데이터 수집 (시뮬레이터 포함)

- 담당: 조원
- 입출력 약속: `docs/data-contract.md` (v1.0)의 2장 inspection 스키마와 이미지 규칙
- 인수 기준: `pytest tests/contract`의 수집 항목 통과
- `spike/` 코드는 import하지 않는다. 필요한 부분은 복사해서 계약 테스트와 함께 새로 넣는다

## 무엇을 만드나

검사 1건 = 128×128 흑백 이미지 + inspection 메타데이터. 이미지는 카메라가 **수직 위에서 내려다본 볼트 6개(2행×3열)의 체결 유무**다.
링 조명을 가정해 그림자를 한쪽으로 두지 않고, 4배 크기로 그린 뒤 128×128로 줄인다.

- 배경: 금속판 (약한 가로 결 + 큰 얼룩)
- 체결(정상): 플랜지 + 육각 머리(방향은 볼트마다 무작위) + 둥근 밝은 윗면
- 미체결(불량): 모따기 고리 + 안쪽으로 어두워지는 나사산 동심원 + 거의 검은 바닥. 불량이면 1~2개가 미체결
- `missing_parts`는 계약 필드 그대로 쓰고, 의미는 **미체결 볼트 번호(1~6)** 로 해석한다
  (계약 문서에는 "누락 부품"이라고 적혀 있다. 표현 차이일 뿐 필드와 규칙은 같다)
- 드리프트 3종: `dark`(어두워짐), `noise`(노이즈 증가), `shift`(위치 이동). 강도는 `collector.toml`

## 사용법

```python
from parts.collector import Collector

c = Collector()                      # 설정: parts/collector/collector.toml
s = c.next_sample()                  # 정상 환경, data_split=pool, 불량은 설정한 불량률로 뽑음
s = c.next_sample(scenario="drift")  # 드리프트 (종류는 무작위, drift_type="dark" 로 지정 가능)
s = c.next_sample(data_split="test_normal", label="NG")  # 라벨 강제 (고정 테스트셋 구성용)

s.inspection   # common/schemas.py 의 Inspection
s.image        # uint8 (128, 128)
```

- 이미지는 `data/<data_split>/INS-000001.png` 로 저장하고, `image_path`는 프로젝트 루트 기준 상대경로다.
- ID는 연속 번호이고, `data/<data_split>/` 에 이미 있는 가장 큰 번호 다음부터 이어진다. 이미 있는 파일은 덮어쓰지 않는다.
- 저장 전·후 검증(128×128 흑백, OK이면 `missing_parts` 비어 있음, 파일이 열리는지)에 실패하면 파일을 남기지 않고 오류를 낸다.
- 같은 시드(기본 42)면 같은 결과가 나온다. 시드는 `Collector(seed=...)`로 바꿀 수 있다.
- 같은 `data/` 폴더에는 한 번에 Collector 하나만 쓴다.

## 설정 파일 이름 변경

처음 정한 `collector.yaml`은 `collector.toml`로 바뀌었다. YAML을 읽는 패키지(PyYAML)가 설치되어 있지 않아서,
Python 표준 라이브러리(`tomllib`)로 읽을 수 있는 TOML을 쓴다. `image_size`와 `part_count`는 계약 고정값(128, 6)이라 다른 값이면 오류가 난다.
