"""frontend/view.py 승인 화면(④-4) 순수 함수 시험. Streamlit·서버·파일 없음."""
from frontend import view
from tests.frontend.test_view_retrain import gate, res, run


def item(n=2, version="v2", status="pending", handled_by=None, reason=None, handled_at=None):
    return {"approval_id": f"APR-{n:06d}", "model_version": version, "requested_by": "system", "status": status, "reason": reason,
            "handled_by": handled_by, "handled_at": handled_at, "created_at": "2026-11-01T08:01:00.000000Z"}


def approvals(*items):
    return res({"items": list(items), "total": len(items)})


def production(version="v1"):
    return res(None if version is None else {"version": version})


def test_pending_and_handled_split_keeps_server_order_and_failure_is_empty():
    data = approvals(item(3, "v3"), item(2, "v2", "approved", "admin"), item(1, "v1", "rejected", "admin", "사유"))
    assert [i["approval_id"] for i in view.pending_items(data)] == ["APR-000003"]
    assert [i["approval_id"] for i in view.handled_items(data)] == ["APR-000002", "APR-000001"]
    failed = view.fail_result(500, "x")
    assert view.pending_items(failed) == [] and view.handled_items(None) == [] and view.find_approval(failed, "APR-000001") is None
    assert view.selected_item(view.pending_items(data), "APR-999999")["approval_id"] == "APR-000003"  # 목록에 없으면 첫 번째
    assert view.selected_item([], None) is None


def test_candidate_run_matches_result_version_and_baseline_rule_mirrors_server():
    runs = res({"items": [run(result="v3", n=3), run(result="v2", n=2), run("failed", "train", n=1)], "total": 3})
    assert view.candidate_run(runs, "v2")["run_id"] == "RUN-000002" and view.candidate_run(runs, "v9") is None
    assert view.candidate_run(view.fail_result(500, "x"), "v2") is None and view.candidate_run(None, "v2") is None
    r = run(gate_result=gate(baseline="v1"))
    assert not view.baseline_mismatch(r, production("v1"))
    assert view.baseline_mismatch(r, production("v2")) and view.baseline_mismatch(r, production(None))
    assert not view.baseline_mismatch(None, production("v2"))  # 학습 기록이 없으면 검사하지 않는다 (서버 규칙)
    assert not view.baseline_mismatch(r, view.fail_result(500, "x"))  # production 을 못 읽었으면 판단하지 않는다
    assert not view.baseline_mismatch(run(gate_result=gate(baseline=None)), production(None))  # 둘 다 없으면 같다
    warning = view.baseline_warning(r, production("v2"))
    assert "v1" in warning and "v2" in warning and "409" in warning and view.baseline_warning(r, production("v1")) is None
    assert view.summary_models("v2", r) == ["v2", "v1"] and view.summary_models("v2", None) == ["v2"]


def test_role_gate_and_reason_rules():
    assert view.can_decide("admin") == (True, None)
    enabled, hint = view.can_decide("operator")
    assert enabled is False and "관리자만" in hint and view.can_decide(None)[0] is False
    assert view.normalize_reason("approve", "  ") is None and view.normalize_reason("approve", " 좋음 ") == "좋음"
    assert view.normalize_reason("reject", "   ") == "" and view.normalize_reason("reject", None) == ""  # 거절은 비어도 그대로 보내 서버 400 에 맡긴다


def test_inflight_marker_blocks_same_request_only_for_fifteen_seconds():
    marker = {"approval_id": "APR-000002", "action": "approve", "ts": 100.0}
    assert view.inflight_active(marker, "APR-000002", 110.0) and not view.inflight_active(marker, "APR-000003", 110.0)
    assert not view.inflight_active(marker, "APR-000002", 100.0 + view.INFLIGHT_SECONDS) and not view.inflight_active(None, "APR-000002", 1.0)
    assert view.inflight_done(marker, approvals(item(2, status="approved"))) and not view.inflight_done(marker, approvals(item(2)))
    assert not view.inflight_done(marker, None) and not view.inflight_done(None, approvals(item(2, status="approved")))


def test_failure_messages_for_each_status():
    text400 = view.failure_message("reject", 400, "reason 은 [필수]입니다")[0]
    assert text400.startswith("거절 사유를 입력해 주세요") and "서버 메시지: reason 은 " + chr(92) + "[필수" + chr(92) + "]입니다" in text400  # 서버 message 를 함께 보이고 무해화한다
    assert view.failure_message("approve", 400, "bad")[0] == "승인 요청 실패 (400): bad"
    assert view.failure_message("approve", 403, "x") == ("이 계정에는 승인 권한이 없습니다.", "error")
    assert view.failure_message("approve", 404, "x")[1] == "warn"
    text, level = view.failure_message("approve", 503, "새 모델 v2 를 불러오지 못해", approvals(item(2)), "APR-000002", production("v1"))
    assert level == "error" and "승인하지 않았습니다" in text and "승인 대기 상태" in text and "운영 모델은 v1" in text and "새 모델 v2" in text
    assert "서버를 다시 읽은 값" not in view.failure_message("approve", 503, "x")[0]  # 다시 읽지 못했으면 사실로 주장하지 않는다
    assert "\\[" in view.failure_message("approve", 500, ":red[x]")[0]  # 서버 문장은 서식 무해화


def test_conflict_is_told_apart_by_rereading_server_state():
    runs = res({"items": [run(result="v2", gate_result=gate(baseline="v1"))], "total": 1})
    done = view.approval_conflict("APR-000002", approvals(item(2, status="approved", handled_by="admin", handled_at="2026-11-01T08:05:00.000000Z")), runs, production("v2"), "m")
    assert done[1] == "info" and "이미 처리된 요청" in done[0] and "승인됨" in done[0] and "admin" in done[0]
    mismatch = view.approval_conflict("APR-000002", approvals(item(2)), runs, production("v3"), "m")
    assert mismatch[1] == "warn" and "기준 모델(v1)" in mismatch[0] and "v3" in mismatch[0]
    other = view.approval_conflict("APR-000002", approvals(item(2)), runs, production("v1"), "다른 사유")
    assert "(409): 다른 사유" in other[0]
    assert "(409)" in view.approval_conflict("APR-000002", None, None, None, "m")[0]


def test_outcome_text_comes_from_reread_values_not_the_response():
    ok = view.sent_outcome("approve", "APR-000002", approvals(item(2, status="approved")), production("v2"))
    assert ok[1] == "ok" and "승인 완료: v2이 운영 모델" in ok[0]
    assert view.sent_outcome("approve", "APR-000002", approvals(item(2, status="approved")), production("v1"))[1] == "warn"
    assert view.sent_outcome("approve", "APR-000002", approvals(item(2)), production("v1"))[1] == "warn"  # 아직 pending 이면 성공이라고 하지 않는다
    assert view.sent_outcome("approve", "APR-000002", None, None)[1] == "warn"
    assert view.sent_outcome("reject", "APR-000002", approvals(item(2, status="rejected")), production("v1"))[0].startswith("거절 완료")
    runs = res({"items": [], "total": 0})
    assert view.action_result("approve", "APR-000002", {"kind": "down"}, None, None, None) == (view.SERVER_DOWN_APPROVAL_TEXT, "warn")
    assert view.action_result("approve", "APR-000002", {"kind": "blocked"}, None, None, None)[1] == "info"
    assert "이미 처리된 요청" in view.action_result("reject", "APR-000002", {"kind": "error", "status": 409, "message": "m"},
                                                 approvals(item(2, status="rejected")), runs, production("v1"))[0]
    assert view.action_result("reject", "APR-000002", {"kind": "error", "status": 403, "message": "m"}, None, None, None)[0].startswith("이 계정")


def test_history_and_pending_tables_sanitize_server_text():
    rows = view.approval_history_rows([item(2, status="rejected", handled_by="admin", reason="a|b :red[x]\n끝", handled_at="2026-11-01T08:05:00.000000Z"),
                                       item(3, status="approved", handled_by="admin")])
    assert rows[0]["상태"] == "거절됨" and rows[1]["사유"] == "없음" and rows[0]["처리자"] == "admin"
    table = view.md_table(view.APPROVAL_HISTORY_COLUMNS, rows)
    assert len(table.splitlines()) == 4 and "\\|" in table and "\\[x\\]" in table
    pending = view.pending_rows([item(2)])
    assert pending[0]["요청"] == "APR-000002" and pending[0]["요청자"] == "system" and set(pending[0]) == set(view.PENDING_COLUMNS)
    assert view.local_datetime_text(None) == "없음" and view.local_datetime_text("깨짐") == "깨짐"
    assert "APR-000002" in view.pending_label(item(2)) and "v2" in view.pending_label(item(2))
    assert view.production_line(production(None)) == "현재 운영 모델: 없음" and "읽지 못했습니다" in view.production_line(view.fail_result(500, "x"))
