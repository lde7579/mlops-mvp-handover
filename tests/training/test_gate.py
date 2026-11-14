"""services/training/gate.py: 승격 게이트 G1~G3 (계약 5장). 순수 함수라 DB·torch 없이 돈다."""
import json

import pytest

from services.training.gate import evaluate_gate
from tests.backend.helpers import make_metrics


def metrics(version="v1", *, drift_acc=0.854, drift_recall=1.0, normal_acc=1.0):
    return make_metrics(
        version,
        normal={"accuracy": normal_acc, "recall": 1.0, "precision": 1.0, "f1": 1.0, "confusion_matrix": [[400, 0], [0, 100]], "n": 500},
        drift={"accuracy": drift_acc, "recall": drift_recall, "precision": 0.8, "f1": 0.9, "confusion_matrix": [[350, 50], [0, 100]], "n": 500})


BASE = metrics("v1")  # drift 정확도 0.854, 재현율 1.0, normal 정확도 1.0


def test_recovered_candidate_passes_all_three_with_the_measured_numbers():
    result = evaluate_gate(BASE, metrics("v2", drift_acc=0.970))  # 3-D 측정 (시드 42): 0.854 -> 0.970
    assert result.passed and result.baseline_version == "v1" and result.failed_ids == []
    assert [c.id for c in result.checks] == ["G1", "G2", "G3"] and all(c.passed for c in result.checks)
    assert result.reasons[0] == "G1 통과: 드리프트 정확도 0.970 ≥ 기존 0.854 + 1%p (기준 0.864)"


def test_first_model_has_no_gate():
    result = evaluate_gate(None, metrics("v1"))
    assert result.passed and result.checks == () and result.baseline_version is None and result.reasons == []
    assert result.to_dict() == {"baseline_version": None, "passed": True, "checks": []}  # 계약 7장 C7 의 모양


def test_g1_boundary_exactly_plus_one_point_passes_even_with_float_error():
    # 0.554 + 0.01 == 0.5640000000000001 이라서 단순 비교(>=)는 정확히 경계인 0.564(500건 중 5건 차이)를 떨어뜨린다
    assert 0.554 + 0.01 > 0.564
    low = metrics("v1", drift_acc=0.554)
    at_edge = evaluate_gate(low, metrics("v2", drift_acc=0.564))
    assert at_edge.checks[0].passed and at_edge.passed
    just_short = evaluate_gate(low, metrics("v2", drift_acc=0.563))
    assert not just_short.checks[0].passed and just_short.failed_ids == ["G1"] and not just_short.passed
    exact = evaluate_gate(BASE, metrics("v2", drift_acc=0.864))  # 0.854 + 0.01 은 오차가 없는 경우
    assert exact.checks[0].passed


def test_g2_boundary_exactly_minus_one_point_passes():
    edge = evaluate_gate(BASE, metrics("v2", drift_acc=0.9, drift_recall=0.99))  # 불량 100건 중 1건 더 놓침
    assert edge.checks[1].passed and edge.passed
    short = evaluate_gate(BASE, metrics("v2", drift_acc=0.9, drift_recall=0.98))
    assert short.failed_ids == ["G2"] and not short.passed


def test_g3_boundary_exactly_minus_one_point_passes():
    edge = evaluate_gate(BASE, metrics("v2", drift_acc=0.9, normal_acc=0.99))  # 500건 중 5건
    assert edge.checks[2].passed and edge.passed
    short = evaluate_gate(BASE, metrics("v2", drift_acc=0.9, normal_acc=0.988))
    assert short.failed_ids == ["G3"] and not short.passed


@pytest.mark.parametrize("over,failed", [
    ({"drift_acc": 0.86}, ["G1"]),
    ({"drift_acc": 0.9, "drift_recall": 0.9}, ["G2"]),
    ({"drift_acc": 0.9, "normal_acc": 0.9}, ["G3"]),
    ({"drift_acc": 0.5, "drift_recall": 0.5, "normal_acc": 0.5}, ["G1", "G2", "G3"]),
])
def test_each_condition_fails_alone_and_all_must_pass(over, failed):
    result = evaluate_gate(BASE, metrics("v2", **over))
    assert result.failed_ids == failed and not result.passed
    assert all(r.startswith(("G1", "G2", "G3")) for r in result.reasons) and sum("미통과" in r for r in result.reasons) == len(failed)


def test_candidate_equal_to_baseline_fails_g1_only():
    assert evaluate_gate(BASE, metrics("v2", drift_acc=0.854)).failed_ids == ["G1"]  # 나아지지 않았다


def test_to_dict_shape_is_json_ready_and_records_the_baseline():
    result = evaluate_gate(metrics("v3", drift_acc=0.9), metrics("v4", drift_acc=0.95))
    data = result.to_dict()
    assert json.loads(json.dumps(data)) == data  # 그대로 JSON 컬럼에 들어간다
    assert data["baseline_version"] == "v3" and data["passed"] is True
    assert set(data["checks"][0]) == {"id", "passed", "candidate", "baseline", "required", "message"}
    assert data["checks"][1]["required"] == ">= 기존 - 1%p" and data["checks"][0]["baseline"] == 0.9
