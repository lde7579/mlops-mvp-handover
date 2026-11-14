"""frontend/view.py 재학습 화면(④-3) 순수 함수 시험. Streamlit·서버·파일 없음."""
import pytest

from frontend import view

CHECKS = [
    {"id": "G1", "passed": True, "candidate": 0.97, "baseline": 0.854, "required": ">= 기존 + 1%p", "message": "G1 통과: 드리프트 정확도 0.970 ≥ 기존 0.854 + 1%p (기준 0.864)"},
    {"id": "G2", "passed": True, "candidate": 1.0, "baseline": 1.0, "required": ">= 기존 - 1%p", "message": "G2 통과: 드리프트 재현율 1.000 ≥ 기존 1.000 - 1%p (기준 0.990)"},
    {"id": "G3", "passed": False, "candidate": 0.95, "baseline": 1.0, "required": ">= 기존 - 1%p", "message": "G3 미통과: 정상 정확도 0.950 < 기존 1.000 - 1%p (기준 0.990)"}]


def gate(passed=True, baseline="v1", checks=None):
    return {"baseline_version": baseline, "passed": passed, "checks": CHECKS if checks is None else checks}


def run(status="succeeded", step="gate", result="v2", gate_result="default", error=None, n=2, started="2026-11-01T08:00:00.000000Z",
        finished="2026-11-01T08:00:41.000000Z"):
    if gate_result == "default":
        gate_result = gate() if status == "succeeded" else None
    return {"run_id": f"RUN-{n:06d}", "status": status, "trigger": "manual", "started_at": started, "finished_at": finished,
            "result_version": result if status == "succeeded" else None, "step": step, "error_message": error, "gate_result": gate_result,
            "train_data_snapshot": {"train": 2000, "pool": 120} if status == "succeeded" else None}


def res(data):
    return view.ok_result(data)


def metrics(acc_normal=1.0, acc_drift=0.854):
    split = lambda acc: {"accuracy": acc, "recall": 1.0, "precision": 0.9, "f1": 0.95, "confusion_matrix": [[400, 0], [0, 100]], "n": 500}
    return {"version": "v1", "normal": split(acc_normal), "drift": split(acc_drift)}


def test_md_cell_blocks_table_and_directive_breakers():
    text = "a|b\nc`d[e]f$g*h_i\\j :red[x]"
    out = view.md_cell(text)
    assert "\n" not in out and "\\|" in out and "\\[" in out and "\\]" in out and "\\$" in out and "\\*" in out and "\\_" in out and "\\`" in out and "\\\\" in out
    assert view.md_cell("정상 문장 12.3%") == "정상 문장 12.3%"
    table = view.md_table(("a", "b"), [{"a": "x|y\nz", "b": ":red[q]"}])
    assert len(table.splitlines()) == 3 and table.splitlines()[2].count("|") - table.splitlines()[2].count("\\|") == 3  # 앞뒤 + 가운데 구분선만 남는다


def test_stages_follow_server_status_and_step():
    labels = lambda r: [(name, mark) for name, mark, _ in view.run_stages(r)]
    assert labels(None) == labels(run("queued", None)) == [("데이터 구성", "대기"), ("학습", "대기"), ("평가", "대기"), ("게이트", "대기")]
    assert labels(run("running", "train")) == [("데이터 구성", "✔ 완료"), ("학습", "진행 중"), ("평가", "대기"), ("게이트", "대기")]
    assert [m for _, m in labels(run("succeeded"))] == ["✔ 완료"] * 4
    assert labels(run("failed", "train")) == [("데이터 구성", "✔ 완료"), ("학습", "✖ 실패"), ("평가", "대기"), ("게이트", "대기")]
    assert [m for _, m in labels(run("failed", None))] == ["대기"] * 4  # 단계 기록 전 실패: 칸 표시 없이 문장만
    assert view.run_status_text(run("running", "data")) == ("실행 중: 데이터 구성", "info")
    assert view.run_status_text(run("succeeded", gate_result=gate(False))) == ("성공 (게이트 미통과)", "warn")
    assert view.run_status_text(None)[1] == "muted" and view.run_status_text(run("failed", "train"))[1] == "error"


def test_outcome_separates_failed_from_gate_failed_and_shows_real_approval_status():
    failed = view.run_outcome(run("failed", "train", error="boom"), None)
    assert failed[1] == "error" and "실패" in failed[0] and "게이트" not in failed[0]
    missed = view.run_outcome(run(gate_result=gate(False)), res({"items": []}))
    assert missed[1] == "warn" and "통과하지 못했습니다" in missed[0] and "승인 요청은 만들어지지 않았고" in missed[0]
    assert view.run_outcome(run("running", "train"), None) is None and view.run_outcome(None, None) is None
    approvals = lambda status: res({"items": [{"approval_id": "APR-000009", "model_version": "v9", "status": "approved"},
                                              {"approval_id": "APR-000002", "model_version": "v2", "status": status}], "total": 2})
    for status, label, level in (("pending", "승인 대기", "info"), ("approved", "승인됨", "ok"), ("rejected", "거절됨", "warn")):
        text, got = view.run_outcome(run(), approvals(status))
        assert (label in text, got) == (True, level) and "APR-000002" in text and "APR-000009" not in text
    assert "찾지 못했습니다" in view.run_outcome(run(), res({"items": []}))[0]
    assert "확인하지 못했습니다" in view.run_outcome(run(), view.fail_result(500, "x"))[0]
    assert "게이트 미적용" in view.run_outcome(run(gate_result=gate(True, baseline=None, checks=[])), None)[0]
    assert "게이트 결과가 없습니다" in view.run_outcome(run(gate_result=None), None)[0]


def test_what_to_read_next():
    assert view.needs_approval_check(run()) and not view.needs_approval_check(run(gate_result=gate(False)))
    assert not view.needs_approval_check(run("failed", "train")) and not view.needs_approval_check(run(gate_result=gate(True, None, [])))
    assert view.models_needed(run()) == ["v2", "v1"] and view.models_needed(run(gate_result=gate(True, None, []))) == ["v2"]
    assert view.models_needed(run("running", "train")) == [] and view.models_needed(None) == []


def test_gate_rows_metric_rows_and_confusion():
    rows = view.gate_rows(gate())
    assert [r["조건"] for r in rows] == ["G1", "G2", "G3"] and rows[0]["후보"] == "97.0%" and rows[2]["결과"] == "✖ 미통과"
    table = view.gate_markdown(rows)
    assert ":green[✔ 통과]" in table and ":red[✖ 미통과]" in table and "G3 미통과" in table and "_passed" not in table
    assert view.gate_rows(None) == [] and view.gate_rows(gate(True, None, [])) == []
    assert view.gate_basis_text(run()) == "게이트가 비교한 기준 모델: v1" and "미적용" in view.gate_basis_text(run(gate_result=gate(True, None, [])))
    assert "게이트 전" in view.gate_basis_text(run("running", "train", gate_result=None))
    cand = metrics(0.99, 0.97)
    rows = view.metric_rows(metrics(), cand)
    assert len(rows) == 8 and rows[0] == {"평가셋": "정상 (test_normal)", "지표": "정확도", "기존": "100.0%", "후보": "99.0%", "차이(%p)": "-1.0"}
    assert [r["차이(%p)"] for r in rows if r["평가셋"].startswith("드리프트") and r["지표"] == "정확도"] == ["+11.6"]
    assert {r["기존"] for r in view.metric_rows(None, cand)} >= {"없음"} and {r["차이(%p)"] for r in view.metric_rows(None, cand)} == {"없음"}
    assert view.confusion_text([[400, 0], [3, 97]]) == "TN 400 · FP 0 · FN 3 · TP 97" and view.confusion_text(None) == "없음"
    assert view.confusion_rows(None, cand)[0]["기존"] == "없음" and "TP 100" in view.confusion_rows(metrics(), cand)[1]["후보"]


def test_history_rows_duration_gate_and_escaped_error():
    items = [run("running", "train", n=5, finished=None, gate_result=None), run("failed", "train", n=4, error="x|y\n:red[z] $1" + "a" * 60),
             run(n=3, gate_result=gate(False)), run(n=2)]
    rows = view.run_history_rows(items)
    assert [r["상태"] for r in rows] == ["실행", "실패", "성공", "성공"] and rows[0]["소요"] == "진행 중" and rows[2]["소요"] == "41초"
    assert [r["게이트"] for r in rows] == ["없음", "없음", "미통과", "통과"] and rows[2]["학습 데이터"] == "train 2000 · pool 120" and rows[1]["학습 데이터"] == "없음"
    assert rows[3]["결과 버전"] == "v2" and rows[1]["결과 버전"] == "없음" and rows[1]["오류"].endswith("…")
    table = view.run_history_markdown(rows).splitlines()
    assert len(table) == 2 + 4 and all("\n" not in line for line in table) and "\\:red" not in table[3] and "\\[z\\]" in table[3] and "\\$1" in table[3]
    assert view.run_history_rows([run("queued", None, finished=None, started=None)])[0]["소요"] == "없음"


def test_conflict_is_told_apart_by_server_state_not_by_message():
    busy, idle = res({"items": [run("running", "train", gate_result=None)], "total": 1}), res({"items": [run()], "total": 1})
    empty, some = res({"items": [], "total": 0}), res({"items": [], "total": 3})
    assert "실행 중" in view.conflict_message(busy, some, "아무 말")[0]
    assert "0건" in view.conflict_message(idle, empty, "아무 말")[0]
    text, level = view.conflict_message(idle, some, "알 수 없는 | 충돌\n[x]")
    assert level == "warn" and "\\|" in text and "\n" not in text
    assert "0건" in view.conflict_message(None, empty, "x")[0] and "알 수 없는" in view.conflict_message(view.fail_result(500, "x"), None, "알 수 없는 충돌")[0]


def test_other_start_errors_and_notes_and_limit_texts():
    assert view.start_failure_message(403, "m")[0] == "이 계정에는 재학습 권한이 없습니다."
    assert "고정 데이터셋이 없어" in view.start_failure_message(503, "고정 데이터셋이 없습니다")[0] and "(500)" in view.start_failure_message(500, "x")[0]
    assert "요청을 보냈지만" in view.SERVER_DOWN_START_TEXT and "다음 갱신" in view.SERVER_DOWN_START_TEXT
    assert "pool 120장" in view.pool_note(run()) and "편입되지 않았습니다" in view.pool_note(run("failed", "train")) and view.pool_note(run("running", "train")) is None
    assert "학습이 읽은 pool만 train으로 편입되고, 학습 중 새로 쌓인 검사는 pool에 남습니다" in view.pool_note(run())  # 실측(직전 120, 학습 중 생성 50, 편입 120, 완료 후 50)
    assert view.snapshot_text(run()).endswith("pool 120장") and "없음" in view.snapshot_text(run("failed", "train"))
    assert view.pool_text(res({"items": [], "total": 7})) == ("7건", "ok") and view.pool_text(res({"items": [], "total": 0}))[1] == "muted"
    assert view.pool_text(view.fail_result(500, "x"))[1] == "error"
    assert "30/90회" in view.LIMIT_POOL_TEXT and "pool 30·60·100·150·200·300장, 90회" in view.LIMIT_POOL_TEXT and "30~300장" not in view.LIMIT_POOL_TEXT and "60장" in view.LIMIT_POOL_TEXT
    assert "evaluate" in view.LIMIT_EVALUATE_TEXT and "[승인] 화면" in view.LIMIT_APPROVAL_TEXT and "관리자" in view.LIMIT_APPROVAL_TEXT and "아직 없" not in view.LIMIT_APPROVAL_TEXT
