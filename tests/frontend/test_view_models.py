"""frontend/view.py 모델 관리 화면(④-5) 순수 함수 시험. Streamlit·서버·파일 없음."""
from frontend import view
from tests.frontend.test_view_retrain import metrics, res

CREATED = "2026-11-01T08:00:00.000000Z"


def model(version, stage, acc_drift=0.9, by=None, at=None):
    return {"version": version, "stage": stage, "metrics": metrics(1.0, acc_drift), "train_data_snapshot": {"train": 2000},
            "created_at": CREATED, "approved_by": by, "approved_at": at}


def models(*items):
    return res({"items": list(items), "total": len(items)})


def standard():
    """v3 운영 중, v2 거절, v1 보관 (최신 먼저)."""
    return models(model("v3", "production", 1.0, "admin", CREATED), model("v2", "rejected", 0.914), model("v1", "archived", 0.854, "admin", CREATED))


def test_production_targets_and_rows_follow_server_list_only():
    data = standard()
    assert view.production_of(data) == "v3" and view.model_versions(data) == ["v3", "v2", "v1"]
    assert view.rollback_targets(data) == ["v1"]  # archived 만. rejected(v2)와 운영 중(v3)은 고를 수 없다
    assert view.rollback_targets(models(model("v1", "production"), model("v2", "staging"), model("v3", "candidate"))) == []
    failed = view.fail_result(500, "x")
    assert view.production_of(failed) is None and view.model_items(failed) == [] and view.rollback_targets(None) == [] and view.find_model(None, "v1") is None
    rows = view.model_rows(view.model_items(data))
    assert [r["단계"] for r in rows] == ["운영 중", "거절됨", "보관(archived)"]
    assert rows[1]["마지막으로 운영에 올린 사람"] == "없음" and rows[1]["올린 시각"] == "없음" and rows[0]["마지막으로 운영에 올린 사람"] == "admin"
    table = view.model_table_markdown(rows)
    assert ":green[운영 중]" in table and ":green[거절됨]" not in table and "_production" not in table


def test_default_compare_is_production_vs_newest_other_and_handles_edges():
    assert view.default_compare(standard()) == ("v3", "v2")
    assert view.default_compare(models(model("v2", "archived"), model("v1", "production"))) == ("v1", "v2")  # 운영 모델이 가장 오래돼도 그 기준
    assert view.default_compare(models(model("v2", "staging"), model("v1", "archived"))) == ("v2", "v1")  # 운영 모델이 없으면 가장 최신
    assert view.default_compare(models(model("v1", "production"))) == ("v1", None)
    assert view.default_compare(models()) == (None, None) and view.default_compare(view.fail_result(500, "x")) == (None, None)


def test_compare_reuses_metric_math_with_version_headers_and_b_minus_a():
    data = standard()
    a, b = view.find_model(data, "v1")["metrics"], view.find_model(data, "v3")["metrics"]
    columns = view.compare_columns("v1", "v3")
    assert columns == ("평가셋", "지표", "v1 (A)", "v3 (B)", "차이(B−A, %p)")
    rows = view.compare_rows("v1", a, "v3", b)
    plain = view.metric_rows(a, b)
    assert len(rows) == len(plain) == 8 and all(set(r) == set(columns) for r in rows)
    drift_accuracy = next(r for r in rows if r["평가셋"].startswith("드리프트") and r["지표"] == "정확도")
    assert drift_accuracy["v1 (A)"] == "85.4%" and drift_accuracy["v3 (B)"] == "100.0%" and drift_accuracy["차이(B−A, %p)"] == "+14.6"
    confusion = view.compare_confusion_rows("v1", a, "v3", b)
    assert view.compare_confusion_columns("v1", "v3") == ("평가셋", "v1 (A)", "v3 (B)")
    assert [set(r) for r in confusion] == [{"평가셋", "v1 (A)", "v3 (B)"}] * 2 and "TN 400" in confusion[0]["v1 (A)"]
    assert view.compare_labels("v1", "v1") == ("v1 (A)", "v1 (B)")  # 같은 버전을 두 칸에 골라도 열 이름이 겹치지 않는다
    single = view.single_metric_rows(b)
    assert len(single) == 8 and set(single[0]) == set(view.SINGLE_METRIC_COLUMNS)


def test_deployment_rows_label_actions_and_missing_reason():
    items = [{"deployment_id": "DEP-000002", "version": "v1", "action": "rollback", "reason": "시연 롤백", "actor": "admin", "at": CREATED},
             {"deployment_id": "DEP-000001", "version": "v3", "action": "deploy", "reason": None, "actor": "admin", "at": CREATED}]
    rows = view.deployment_rows(items)
    assert [r["종류"] for r in rows] == ["롤백", "배포"] and rows[0]["사유"] == "시연 롤백" and rows[1]["사유"] == "없음"
    assert set(rows[0]) == set(view.DEPLOYMENT_COLUMNS) and view.deployment_items(view.fail_result(500, "x")) == []
    assert view.DEPLOYMENT_LIMIT == 20


def test_only_admin_can_roll_back_and_reason_is_sent_as_stripped_even_if_empty():
    assert view.can_rollback("admin") == (True, None)
    assert view.can_rollback("operator") == (False, view.ROLLBACK_PERMISSION_TEXT) and view.can_rollback(None)[0] is False
    assert "관리자만" in view.ROLLBACK_PERMISSION_TEXT
    assert view.normalize_rollback_reason("  시연 롤백 ") == "시연 롤백" and view.normalize_rollback_reason("   ") == "" and view.normalize_rollback_reason(None) == ""


def test_inflight_marker_is_per_version_and_expires_after_15_seconds():
    marker = {"version": "v1", "ts": 100.0}
    assert view.rollback_inflight_active(marker, "v1", 110.0) and not view.rollback_inflight_active(marker, "v3", 110.0)
    assert not view.rollback_inflight_active(marker, "v1", 100.0 + view.INFLIGHT_SECONDS) and not view.rollback_inflight_active(None, "v1", 1.0)
    assert not view.rollback_inflight_active(marker, "v1", 50.0)  # 시계가 거꾸로 가면 영구 잠금이 되지 않도록 무시
    assert view.rollback_inflight_done(marker, standard()) is False  # v1 은 아직 archived
    assert view.rollback_inflight_done(marker, models(model("v1", "production"))) is True
    assert view.rollback_inflight_done(None, standard()) is False and view.rollback_inflight_done(marker, view.fail_result(500, "x")) is False


def test_failure_messages_cover_each_status_with_server_text():
    text, level = view.rollback_failure_message(400, "reason 은 필수입니다")
    assert level == "error" and "사유" in text and "500자" in text and "reason 은 필수입니다" in text
    assert view.rollback_failure_message(403, "x") == ("이 계정에는 롤백 권한이 없습니다.", "error")
    text, level = view.rollback_failure_message(404, "x")
    assert level == "warn" and "찾을 수 없습니다" in text and "다시 읽었습니다" in text
    text, level = view.rollback_failure_message(503, "모델 v1 을 불러오지 못해", standard(), "v1")
    assert level == "error" and "기존 운영 모델을 계속 사용합니다" in text and "운영 모델은 v3" in text and "모델 v1 을 불러오지 못해" in text
    assert "서버를 다시 읽은 값" not in view.rollback_failure_message(503, "m", view.fail_result(500, "x"), "v1")[0]  # 다시 읽지 못했으면 값을 지어내지 않는다
    assert "(500)" in view.rollback_failure_message(500, "boom")[0]


def test_conflict_is_told_apart_by_reread_stage_not_by_message():
    text, level = view.rollback_conflict("v3", standard(), "아무 글자")
    assert level == "info" and "이미 운영 중" in text
    text, level = view.rollback_conflict("v2", standard(), "아무 글자")
    assert level == "warn" and "거절됨 상태라서 롤백할 수 없습니다" in text
    text, level = view.rollback_conflict("v9", standard(), "원문")
    assert level == "warn" and "409" in text and "원문" in text
    assert view.rollback_conflict("v1", view.fail_result(500, "x"), "원문")[0].endswith("원문")


def test_outcome_comes_from_reread_server_list_not_from_response():
    assert view.rollback_outcome("v3", standard()) == ("롤백 완료: v3이 운영 모델입니다 (서버에서 다시 읽은 값).", "ok")
    text, level = view.rollback_outcome("v1", standard())
    assert level == "warn" and "보관(archived) 상태" in text and "운영 모델: v3" in text
    assert view.rollback_outcome("v9", standard())[1] == "warn" and view.rollback_outcome("v1", None)[1] == "warn"


def test_result_dispatch_by_kind():
    data = standard()
    assert view.rollback_result("v1", {"kind": "blocked"}, data) == (view.ROLLBACK_BLOCKED_TEXT, "info")
    assert view.rollback_result("v1", {"kind": "down"}, None) == (view.SERVER_DOWN_ROLLBACK_TEXT, "warn")
    assert "이미 운영 중" in view.rollback_result("v3", {"kind": "error", "status": 409, "message": "m"}, data)[0]
    assert view.rollback_result("v1", {"kind": "error", "status": 403, "message": "m"}, data)[0].startswith("이 계정에는")
    assert view.rollback_result("v3", {"kind": "sent"}, data)[1] == "ok"
    assert "롤백" in view.SERVER_DOWN_ROLLBACK_TEXT or "응답" in view.SERVER_DOWN_ROLLBACK_TEXT
