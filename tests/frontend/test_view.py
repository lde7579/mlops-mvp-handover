"""frontend/view.py 테스트: 화면 문장·서식을 만드는 순수 함수 (Streamlit 불필요)."""
from frontend import config, view


def health(status="ok", database="ok", version="v1", loaded=True):
    return {"status": status, "database": database, "model": {"production_version": version, "loaded": loaded}}


def test_role_label():
    assert view.role_label("operator") == "운영자" and view.role_label("admin") == "관리자"
    assert view.role_label(None) == "알 수 없음" and view.role_label("guest") == "guest"
    assert view.session_text("admin", "admin") == "admin (관리자)"


def test_backend_status_levels():
    assert view.backend_status(health()) == ("정상", "ok")
    assert view.backend_status(None) == ("연결 안 됨", "error")
    assert view.backend_status(health(database="error", status="degraded"))[1] == "error"
    assert view.backend_status(health(status="degraded"))[1] == "warn"


def test_model_status_cases():
    assert view.model_status(health()) == ("v1 (불러옴)", "ok")
    assert view.model_status(health(loaded=False)) == ("v1 (아직 불러오지 못함)", "warn")
    assert view.model_status(health(version=None, loaded=False)) == ("운영 모델 없음", "warn")
    assert view.model_status(None) == ("확인 불가", "error")


def test_login_error_messages():
    assert "연결할 수 없습니다 (http://x)" in view.login_error_message(None, True, "http://x")
    assert view.login_error_message(401, False, "u") == "계정 또는 비밀번호가 틀렸습니다."
    assert "확인" in view.login_error_message(400, False, "u")
    assert "503" in view.login_error_message(503, False, "u")


def test_is_expired():
    assert view.is_expired(None, 100.0)
    assert view.is_expired(100.0, 100.0) and view.is_expired(99.0, 100.0)
    assert not view.is_expired(101.0, 100.0)


def test_config_defaults_and_override(monkeypatch):
    monkeypatch.delenv(config.API_URL_ENV, raising=False)
    assert config.api_url() == "http://127.0.0.1:8000"
    monkeypatch.setenv(config.API_URL_ENV, " http://192.168.0.5:9000/ ")
    assert config.api_url() == "http://192.168.0.5:9000"
    monkeypatch.setenv(config.API_URL_ENV, "  ")
    assert config.api_url() == config.DEFAULT_API_URL
    assert 3 <= config.refresh_seconds() <= 5          # api-v1.md 1장: 3~5초 폴링
    assert config.CONNECT_TIMEOUT_SECONDS <= 1.0        # 서버가 꺼져도 화면이 오래 멈추지 않게


# ---------- 대시보드 (④-1b) ----------
from datetime import datetime, timedelta, timezone  # noqa: E402

ok, fail = view.ok_result, view.fail_result


def warning(active=None, n=0, accuracy=None, version="v1", size=30, threshold=0.95, confirmed=True):
    return ok({"performance_warning": {"model_version": version, "active": active, "n": n, "accuracy": accuracy,
                                       "recent_n": size, "accuracy_threshold": threshold, "confirmed": confirmed}})


def deployments(version, action="deploy"):
    return ok({"items": [{"deployment_id": "DEP-1", "version": version, "action": action, "reason": None, "actor": "admin", "at": "x"}], "total": 1})


def test_today_start_keeps_timezone_and_resets_time():
    kst = timezone(timedelta(hours=9))
    assert view.today_start(datetime(2026, 11, 1, 23, 59, 59, 999999, tzinfo=kst)) == "2026-11-01T00:00:00+09:00"
    assert view.today_start(datetime(2026, 11, 2, 0, 0, 0, tzinfo=kst)) == "2026-11-02T00:00:00+09:00"


def test_warning_red_green_and_confirmed_flag():
    text, level = view.performance_warning(warning(True, 30, 0.8), None)
    assert level == "error" and "성능 저하 경고" in text and "80.0%" in text and "95.0%" in text
    text, level = view.performance_warning(warning(False, 30, 0.97), None)
    assert level == "ok" and text.startswith("정상") and "97.0%" in text
    assert "(권장값)" not in text
    assert "(권장값)" in view.performance_warning(warning(False, 30, 0.97, confirmed=False), None)[0]


def test_warning_hold_without_production_model():
    text, level = view.performance_warning(warning(None, 0, None, version=None), None)
    assert level == "muted" and "운영 모델이 없어" in text


def test_warning_hold_under_30_without_deploy_record_shows_count():
    text, level = view.performance_warning(warning(None, 12), ok({"items": [], "total": 0}))
    assert level == "muted" and "12/30건" in text and "방금 모델이" not in text
    assert "12/30건" in view.performance_warning(warning(None, 12), None)[0]  # 배포 기록을 읽지 못해도 기본 문구


def test_warning_hold_right_after_model_swap_and_after_rollback():
    text, level = view.performance_warning(warning(None, 5, version="v2"), deployments("v2"))
    assert level == "muted" and "방금 모델이 v2으로 바뀌었습니다" in text and "30건" in text
    text, _ = view.performance_warning(warning(None, 0, version="v1"), deployments("v1", action="rollback"))  # 롤백으로 v1 복귀
    assert "방금 모델이 v1으로 바뀌었습니다" in text
    text, _ = view.performance_warning(warning(None, 5, version="v1"), deployments("v2"))  # 최신 배포 버전이 다르면 교체 직후가 아니다
    assert "방금 모델이" not in text and "5/30건" in text


def test_no_hold_text_once_30_judgments_after_swap():
    for drift in (warning(False, 30, 0.99, version="v2"), warning(True, 30, 0.5, version="v2")):
        text, level = view.performance_warning(drift, deployments("v2"))
        assert level in ("ok", "error") and "판단 보류" not in text and "방금 모델이" not in text
    assert not view.needs_deployment_check(warning(False, 30, 0.99, version="v2"))  # 이때는 배포 기록을 읽지도 않는다
    assert view.needs_deployment_check(warning(None, 5, version="v2"))
    assert not view.needs_deployment_check(warning(None, 0, version=None))
    assert not view.needs_deployment_check(fail(500, "x"))


def test_warning_when_drift_call_failed_is_distinct_from_hold():
    text, level = view.performance_warning(fail(500, "서버 오류"), None)
    assert level == "muted" and "확인할 수 없습니다" in text and "판단 보류" not in text


def perf(n=30, accuracy=0.9, recall=0.8, size=30):
    return ok({"criteria": {"recent_n": size, "accuracy_threshold": 0.95, "confirmed": True},
               "recent": {"model_version": "v1", "n": n, "accuracy": accuracy, "recall": recall, "confusion_matrix": [[1, 0], [0, 1]]}})


def test_cards_production_and_totals():
    assert view.card_production(ok({"version": "v2"}))["value"] == "v2"
    none = view.card_production(ok(None))
    assert none["value"] == "운영 모델 없음" and none["level"] == "warn"
    assert view.card_production(fail(500, "boom")) == {"label": "운영 모델", "value": "조회 실패", "level": "error", "caption": "boom"}
    assert view.card_today(ok({"total": 0}))["value"] == "0건"
    assert view.card_pool(ok({"total": 41}))["value"] == "41건"
    assert view.card_pending(ok({"total": 0}))["level"] == "ok" and view.card_pending(ok({"total": 1}))["level"] == "warn"
    assert view.card_pool(fail(None, "x"))["value"] == "조회 실패"


def test_card_ng_rate_null_zero_and_value():
    def stats(rate, total=0, ng=0):
        return ok({"total": total, "ng_count": ng, "ng_rate": rate, "recent": {}})

    assert view.card_ng_rate(stats(None))["value"] == "판정 없음"
    zero = view.card_ng_rate(stats(0.0, 10, 0))
    assert zero["value"] == "0.0%" and zero["level"] == "ok"  # 0% 와 "판정 없음"은 다르다
    card = view.card_ng_rate(stats(0.2, 50, 10))
    assert card["value"] == "20.0%" and "전체" in card["label"] and "10건" in card["caption"]


def test_card_accuracy_counts_nulls_and_small_n():
    full = view.card_accuracy(perf())
    assert full["value"] == "90.0%" and "재현율 80.0%" in full["caption"] and "최근 30건" in full["caption"]
    assert "7/30건" in view.card_accuracy(perf(n=7))["caption"]
    assert "해당 없음" in view.card_accuracy(perf(recall=None))["caption"]
    empty = view.card_accuracy(perf(n=0, accuracy=None, recall=None))
    assert empty["value"] == "판정 없음" and "0/30건" in empty["caption"]
    assert view.card_accuracy(fail(500, "x"))["value"] == "조회 실패"


def snapshot(**over):
    snap = {"health": ok({"status": "ok", "database": "ok", "model": {"production_version": "v1", "loaded": True}}),
            "simulator": ok({"running": True, "drift": {"enabled": False, "drift_type": None}, "last_error": None}),
            "training": ok({"items": [], "total": 0}), "production": ok({"version": "v1"}), "today": ok({"total": 3}),
            "pool": ok({"total": 2}), "stats": ok({"total": 3, "ng_count": 1, "ng_rate": 0.33, "recent": {}}),
            "performance": perf(), "approvals": ok({"total": 0}), "drift": warning(False, 30, 0.97), "deployments": None}
    snap.update(over)
    return snap


def test_stage_collect_cases():
    def sim(running, drift=None, err=None):
        return ok({"running": running, "last_error": err, "drift": {"enabled": drift is not None, "drift_type": drift}})

    assert view.stage_collect(sim(True)) == ("동작중", "ok")
    assert view.stage_collect(sim(False)) == ("정지", "muted")
    assert view.stage_collect(sim(True, "dark")) == ("동작중 · 드리프트 주입 중: 어두움", "warn")
    assert "노이즈" in view.stage_collect(sim(True, "noise"))[0] and "이동" in view.stage_collect(sim(True, "shift"))[0]
    text, level = view.stage_collect(sim(False, err="생성 실패"))
    assert level == "error" and "스스로 멈춤: 생성 실패" in text
    assert view.stage_collect(fail(500, "x")) == ("확인 불가", "error")


def test_stage_train_cases():
    def run(**kw):
        base = {"status": "succeeded", "step": None, "result_version": "v2", "gate_result": {"passed": True}}
        return ok({"items": [dict(base, **kw)], "total": 1})

    assert view.stage_train(ok({"items": [], "total": 0}))[0].startswith("대기")
    assert view.stage_train(run(status="queued")) == ("실행 대기 중", "info")
    for step, label in (("data", "데이터 구성"), ("train", "학습"), ("evaluate", "평가"), ("gate", "게이트")):
        assert view.stage_train(run(status="running", step=step)) == (f"실행 중: {label}", "info")
    assert view.stage_train(run()) == ("성공 (v2)", "ok")
    assert view.stage_train(run(gate_result={"passed": False})) == ("성공 (v2), 게이트 미통과", "warn")
    assert view.stage_train(run(status="failed", result_version=None, gate_result=None)) == ("실패", "error")


def test_stages_line_order_and_deploy_and_infer():
    assert [s[0] for s in view.stages(snapshot())] == ["수집", "추론", "학습", "배포"]
    assert view.stage_deploy(ok(None)) == ("운영 모델 없음", "warn") and view.stage_deploy(ok({"version": "v2"})) == ("v2", "ok")
    assert view.stage_infer(fail(None, "x")) == ("확인 불가", "error")
    assert view.stage_infer(snapshot()["health"]) == ("v1 (불러옴)", "ok")


def test_cards_returns_six_and_unauthorized_detection():
    assert [c["label"] for c in view.cards(snapshot())] == [
        "운영 모델", "오늘 검사 건수", "불량률 (판정 기준, 전체)", "최근 정확도 · 재현율", "승인 대기", "pool 건수"]
    assert not view.has_unauthorized(snapshot())
    assert view.has_unauthorized(snapshot(approvals=fail(401, "만료")))


def test_clock_and_status_texts():
    assert view.last_success_text(None, 4).startswith("아직 성공한 갱신이 없습니다")
    assert "마지막 갱신 성공" in view.last_success_text(0.0, 4) and "4초마다" in view.last_success_text(0.0, 4)
    assert "http://x" in view.server_down_text("http://x")
    assert 3 <= config.refresh_seconds() <= 5


# ---------- 검사/추론 화면 (④-2) ----------
from datetime import datetime, timedelta, timezone  # noqa: E402


def pred(true="NG", predicted="NG", score=0.93, ms=120, iid="INS-000002", at="2026-11-01T08:00:05.100000Z", ver="v1"):
    return {"prediction_id": "PRD-1", "inspection_id": iid, "model_version": ver, "predicted_label": predicted, "score": score,
            "threshold": 0.5, "latency_ms": ms, "predicted_at": at, "true_label": true, "correct": true == predicted}


def test_misjudgement_kind_four_combinations():
    assert view.misjudgement_kind("NG", "NG") is None and view.misjudgement_kind("OK", "OK") is None
    assert view.misjudgement_kind("NG", "OK") == "missed"  # 불량을 놓침
    assert view.misjudgement_kind("OK", "NG") == "false_alarm"  # 정상을 불량으로


def test_match_text_detail_and_row_style():
    assert [view.match_text(k) for k in (None, "missed", "false_alarm")] == ["✔", "✖ 미탐", "✖ 오경보"]
    assert view.match_detail("NG", None) == "정답: NG — 일치 ✔"
    assert "미탐" in view.match_detail("NG", "missed") and "오경보" in view.match_detail("OK", "false_alarm")
    assert view.row_style(None) == ""
    assert view.row_style("missed") != view.row_style("false_alarm") != "" and "color" in view.row_style("missed")  # 배경과 글자색을 함께 정한다


def test_history_rows_convert_fields():
    rows = view.history_rows([pred(), pred(true="NG", predicted="OK", score=0.2, iid="INS-000003"), pred(true="OK", predicted="NG", iid="INS-000004")])
    first = rows[0]
    assert set(first) == {"시각", "검사번호", "모델", "판정", "점수", "정답", "정오", "처리(ms)", "_kind"}
    assert (first["검사번호"], first["모델"], first["판정"], first["점수"], first["정답"], first["정오"], first["처리(ms)"], first["_kind"]) == \
        ("INS-000002", "v1", "NG", "0.93", "NG", "✔", 120, None)
    assert len(first["시각"]) == 8 and first["시각"][2] == ":"  # HH:MM:SS (현지 시각)
    assert [r["정오"] for r in rows] == ["✔", "✖ 미탐", "✖ 오경보"] and [r["_kind"] for r in rows] == [None, "missed", "false_alarm"]
    assert view.history_rows([]) == []


def test_history_markdown_highlights_misjudged_rows_without_pandas():
    rows = view.history_rows([pred(), pred(true="NG", predicted="OK", ms=5200, iid="INS-000003"), pred(true="OK", predicted="NG", iid="INS-000004"),
                              pred(true="OK", predicted="OK", ms=5200, iid="INS-000005")])
    lines = view.history_markdown(rows).splitlines()
    assert lines[0] == "| 시각 | 검사번호 | 모델 | 판정 | 점수 | 정답 | 정오 | 처리(ms) |" and lines[1] == "|---|---|---|---|---|---|---|---|"
    assert len(lines) == 2 + 4 and ":red[**NG**]" in lines[2] and "background" not in lines[2]  # 정판 NG: 글자만 빨강
    assert all(c.startswith(":red-background[") for c in lines[3].strip("| ").split(" | ")) and ":red-background[5200]" in lines[3]  # 미탐: 모든 칸 빨간 배경
    assert all(c.startswith(":orange-background[") for c in lines[4].strip("| ").split(" | ")) and ":red-background[:red[**NG**]]" not in lines[4]
    assert ":red[5200]" in lines[5] and "background" not in lines[5]  # 정판이어도 5000 ms 초과는 빨간 글자
    assert view.history_markdown([]).count("\n") == 1  # 머리만


def test_ng_summary_text():
    items = [pred(), pred(true="NG", predicted="OK"), pred(true="OK", predicted="NG"), pred(true="OK", predicted="OK")]
    assert view.ng_summary_text(items) == "최근 4건 중 NG 판정 2건, 오판 2건 (미탐 1, 오경보 1)"
    assert view.ng_summary_text([pred(true="OK", predicted="OK")]) == "최근 1건 중 NG 판정 0건, 오판 0건 (미탐 0, 오경보 0)"
    assert view.ng_summary_text([]) == "아직 판정이 없습니다."


def test_latency_text_boundary_at_5000():
    assert view.latency_text(142) == ("서버 처리 142 ms (기준 5000 ms 이내 ✔)", "ok")
    assert view.latency_text(5000)[1] == "ok" and view.latency_text(5001)[1] == "error" and "초과" in view.latency_text(5001)[0]


def test_elapsed_text_boundary_and_wording():
    text, level = view.elapsed_text(1.34)
    assert level == "ok" and text.startswith("이번 수동 판정: 생성→판정→이미지 받기까지 1.3초") and "5초 이내 ✔" in text and "브라우저" in text
    assert view.elapsed_text(5.0)[1] == "ok" and view.elapsed_text(5.01) [1] == "error" and "5초 초과 ✖" in view.elapsed_text(5.01)[0]


def test_seconds_between_and_display_delay_text():
    later = datetime(2026, 11, 1, 8, 0, 3, 400000, tzinfo=timezone.utc)
    assert abs(view.seconds_between("2026-11-01T08:00:00Z", later) - 3.4) < 1e-6
    assert view.seconds_between(None, later) is None and view.seconds_between("이상한 값", later) is None
    assert view.seconds_between("2026-11-01T08:00:10Z", later) < 0
    assert "약 3.4초" in view.display_delay_text(3.4) and "참고값" in view.display_delay_text(3.4) and "5초를 넘을 수" in view.display_delay_text(3.4)
    assert "측정 안 함" in view.display_delay_text(None) and "측정 안 함" in view.display_delay_text(-1.0)
    assert "✔" not in view.display_delay_text(3.4) and "✖" not in view.display_delay_text(9.9)  # 합격/불합격 표시를 하지 않는다


def test_refresh_note_matches_config():
    assert view.REFRESH_NOTE == f"{config.refresh_seconds()}초"


def status(running=True, drift=(False, None), last_error=None):
    return {"running": running, "interval_seconds": 2.0 if running else None, "defect_rate": 0.2 if running else None,
            "drift": {"enabled": drift[0], "drift_type": drift[1]}, "generated": 12, "judged": 11, "judge_failures": 1,
            "last_error": last_error, "started_at": None}


def test_simulator_detail_text():
    text, level = view.simulator_detail_text(status())
    assert text == "동작중 · 간격 2.0초 · 불량률 20% · 드리프트: 꺼짐 · 만든 12건 · 판정 11건 · 건너뜀 1건" and level == "ok"
    text, level = view.simulator_detail_text(status(drift=(True, "dark")))
    assert "드리프트 주입: 어두움" in text and level == "warn"
    text, level = view.simulator_detail_text(status(running=False, last_error="연속 실패"))
    assert text.startswith("정지 · 드리프트: 꺼짐") and "마지막 오류: 연속 실패" in text and level == "error"
    assert view.simulator_detail_text(status(running=False))[1] == "muted"


def test_action_messages():
    assert view.action_message("start", 409, "이미 실행 중입니다") == ("이미 실행 중입니다. 현재 상태를 다시 확인했습니다.", "info")
    assert view.action_message("stop", 403, "x") == ("이 계정에는 권한이 없습니다.", "error")
    text, level = view.action_message("drift_on", 503, "운영 모델이 없음")
    assert "드리프트 켜기 요청 실패 (503)" in text and "운영 모델이 없음" in text and level == "error"
    assert "만들었지만 판정하지 못했습니다" in view.manual_failure_text("INS-000007", 503, "운영 중인 모델이 없습니다")
    assert "INS-000007" in view.manual_failure_text("INS-000007", 503, "x")
    assert view.manual_failure_text(None, 400, "x").startswith("1건 생성에 실패했습니다")
