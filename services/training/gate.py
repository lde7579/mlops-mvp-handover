"""승격 게이트 G1~G3 판정 (docs/data-contract.md 5장). DB, 파일, torch 를 쓰지 않는 순수 함수다.

입력은 두 모델의 metrics 뿐이고, 출력은 통과 여부와 조건별 근거 문장이다 (근거 수치는 C6 측정값이 아니라 실제 입력값).
- G1  test_drift 정확도  후보 >= 기존 + 1%p   (드리프트 환경에서 실제로 나아졌는가)
- G2  test_drift 재현율  후보 >= 기존 - 1%p   (불량을 더 놓치지 않는가)
- G3  test_normal 정확도 후보 >= 기존 - 1%p   (정상 환경 성능이 깎이지 않았는가)
현재 production 이 없는 첫 모델에는 게이트를 적용하지 않는다 (통과, 조건 없음).

비교는 1e-9 의 오차를 허용한다. 0.854 + 0.01 은 컴퓨터에서 0.8640000000000001 이 되므로, 정확히 경계인 값(500건 중 5건 차이)이
부동소수점 오차 때문에 불합격이 되지 않게 하기 위해서다.
"""
from dataclasses import dataclass
from typing import Optional

from common.schemas import ModelMetrics

MARGIN = 0.01   # 1%p
TOLERANCE = 1e-9


@dataclass(frozen=True)
class GateCheck:
    id: str
    passed: bool
    candidate: float      # 후보의 값
    baseline: float       # 기존 production 의 값
    required: str         # 요구 조건 (예: ">= 기존 + 1%p")
    message: str          # 근거 문장

    def to_dict(self) -> dict:
        return {"id": self.id, "passed": self.passed, "candidate": self.candidate, "baseline": self.baseline,
                "required": self.required, "message": self.message}


@dataclass(frozen=True)
class GateResult:
    baseline_version: Optional[str]   # 비교 기준이 된 운영 버전 (첫 모델이면 None)
    passed: bool
    checks: tuple[GateCheck, ...]

    @property
    def reasons(self) -> list[str]:
        """조건별 근거 문장 (게이트 결과 화면과 감사 기록에 쓴다)."""
        return [c.message for c in self.checks]

    @property
    def failed_ids(self) -> list[str]:
        return [c.id for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        """training_run.gate_result 에 저장하는 JSON (계약 7장 C7)."""
        return {"baseline_version": self.baseline_version, "passed": self.passed,
                "checks": [c.to_dict() for c in self.checks]}


def _check(check_id: str, label: str, candidate: float, baseline: float, delta: float) -> GateCheck:
    threshold = baseline + delta
    passed = candidate >= threshold - TOLERANCE
    sign = "+" if delta > 0 else "-"
    required = f">= 기존 {sign} 1%p"
    verdict = "통과" if passed else "미통과"
    compare = "≥" if passed else "<"
    message = (f"{check_id} {verdict}: {label} {candidate:.3f} {compare} 기존 {baseline:.3f} {sign} 1%p "
               f"(기준 {threshold:.3f})")
    return GateCheck(check_id, passed, candidate, baseline, required, message)


def evaluate_gate(current: Optional[ModelMetrics], candidate: ModelMetrics) -> GateResult:
    """현재 production 의 metrics(없으면 None)와 후보의 metrics 로 G1~G3 을 판정한다."""
    if current is None:  # 첫 모델: 게이트 미적용
        return GateResult(baseline_version=None, passed=True, checks=())
    checks = (
        _check("G1", "드리프트 정확도", candidate.drift.accuracy, current.drift.accuracy, +MARGIN),
        _check("G2", "드리프트 재현율", candidate.drift.recall, current.drift.recall, -MARGIN),
        _check("G3", "정상 정확도", candidate.normal.accuracy, current.normal.accuracy, -MARGIN),
    )
    return GateResult(baseline_version=current.version, passed=all(c.passed for c in checks), checks=checks)
