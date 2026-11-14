"""frontend/pages/inspection.py 화면 시험: streamlit.testing 의 AppTest + 가짜 ApiClient (실제 서버·네트워크·data/·models/ 없음)."""
import io
import time
from datetime import datetime, timezone

import pytest
from PIL import Image

import frontend.api_client as api_client
from frontend import state
from frontend.pages import inspection

NOW = datetime(2026, 11, 1, 8, 0, 3, 400000, tzinfo=timezone.utc)  # 화면이 새 판정을 읽는 시각 (고정)


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("L", (128, 128), 128).save(buffer, format="PNG")
    return buffer.getvalue()


PNG = png_bytes()


def prediction(n, predicted="NG", true="NG", ms=120):
    return {"prediction_id": f"PRD-{n:06d}", "inspection_id": f"INS-{n:06d}", "model_version": "v1", "predicted_label": predicted,
            "score": 0.93 if predicted == "NG" else 0.12, "threshold": 0.5, "latency_ms": ms,
            "predicted_at": f"2026-11-01T08:00:{n:02d}.100000Z", "true_label": true, "correct": predicted == true}


def stopped():
    return {"running": False, "interval_seconds": None, "defect_rate": None, "drift": {"enabled": False, "drift_type": None},
            "generated": 0, "judged": 0, "judge_failures": 0, "last_error": None, "started_at": None}


def running():
    return {**stopped(), "running": True, "interval_seconds": 2.0, "defect_rate": 0.2, "generated": 5, "judged": 5}


class FakeApi:
    """api_client.ApiClient 를 대신한다. world 가 '서버의 실제 상태'이고, fail[이름] 에 예외를 넣으면 그 호출이 실패한다."""
    world = {}
    calls = []
    args = {}
    fail = {}

    def __init__(self, base_url=None, token=None, client=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def _do(self, name, **args):
        FakeApi.calls.append(name)
        FakeApi.args.setdefault(name, []).append(args)
        if name in FakeApi.fail:
            raise FakeApi.fail[name]

    def simulator_status(self):
        self._do("status")
        return FakeApi.world["status"]

    def predictions(self, limit=20):
        self._do("predictions", limit=limit)
        items = FakeApi.world["items"]
        return {"items": items[:limit], "total": len(items)}

    def inspection(self, inspection_id):
        self._do("inspection", id=inspection_id)
        return {"inspection_id": inspection_id, "captured_at": "2026-11-01T08:00:00.000000Z", "predictions": []}

    def inspection_image(self, inspection_id):
        self._do("image", id=inspection_id)
        return PNG

    def simulator_start(self, defect_rate, interval_seconds):
        self._do("start", defect_rate=defect_rate, interval_seconds=interval_seconds)
        FakeApi.world["status"] = {**running(), "defect_rate": defect_rate, "interval_seconds": interval_seconds}
        return {"running": False}  # 일부러 틀린 본문: 화면은 응답이 아니라 GET /simulator/status 를 믿어야 한다

    def simulator_stop(self):
        self._do("stop")
        FakeApi.world["status"] = stopped()
        return {"running": True}  # 일부러 틀린 본문

    def simulator_drift(self, enabled, drift_type=None):
        self._do("drift", enabled=enabled, drift_type=drift_type)
        FakeApi.world["status"] = {**FakeApi.world["status"], "drift": {"enabled": enabled, "drift_type": drift_type}}
        return {}

    def simulator_generate(self, scenario="normal"):
        self._do("generate", scenario=scenario)
        n = len(FakeApi.world["items"]) + 1
        FakeApi.world["new"] = n
        return {"inspection_id": f"INS-{n:06d}", "captured_at": "2026-11-01T08:00:10.000000Z"}

    def predict(self, inspection_id):
        self._do("predict", id=inspection_id)
        FakeApi.world["items"] = [prediction(FakeApi.world["new"])] + FakeApi.world["items"]
        return FakeApi.world["items"][0]


@pytest.fixture
def fake(monkeypatch):
    FakeApi.world = {"status": running(), "items": [prediction(3), prediction(2, "OK", "NG"), prediction(1, "NG", "OK")]}
    FakeApi.calls, FakeApi.args, FakeApi.fail = [], {}, {}
    monkeypatch.setattr(api_client, "ApiClient", FakeApi)
    monkeypatch.setattr(inspection, "_now", lambda: NOW)
    ticks = iter([10.0, 11.34, 20.0, 25.6])  # 수동 클릭 첫 번째 1.34초, 두 번째 5.6초
    monkeypatch.setattr(inspection, "_tick", lambda: next(ticks))
    return FakeApi


def entry():
    from frontend.pages import inspection as page

    page.render()


def new_page(logged_in=True):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(entry, default_timeout=15)
    if logged_in:
        at.session_state[state.AUTH_KEY] = {"token": "tok", "role": "operator", "username": "operator", "expires_at": time.time() + 3600}
    return at


def texts(elements):
    return " | ".join(str(e.value) for e in elements)


def click(at, key):
    at.button(key=key).click().run()
    assert not at.exception


def test_logged_out_makes_no_calls(fake):
    at = new_page(logged_in=False).run()
    assert not at.exception and "로그인해 주세요" in texts(at.info) and fake.calls == []


def test_first_draw_shows_status_latest_ng_and_history(fake):
    at = new_page().run()
    assert not at.exception
    body = texts(at.markdown)
    assert "동작중 · 간격 2.0초 · 불량률 20%" in body and "드리프트: 꺼짐" in body
    assert "# :red[NG]" in body and "검사 INS-000003" in body and "점수 0.93 (기준 0.50) · 모델 v1" in body
    assert "서버 처리 120 ms (기준 5000 ms 이내 ✔)" in body and "정답: NG — 일치 ✔" in body
    assert len(at.get("image")) == 1  # 최신 이미지
    assert "최근 3건 중 NG 판정 2건, 오판 2건 (미탐 1, 오경보 1)" in texts(at.caption) and "전체 3건 중 최근 3건" in texts(at.caption)
    table = [m.value for m in at.markdown if str(m.value).startswith("| 시각")][0].splitlines()
    assert len(table) == 2 + 3 and "_kind" not in table[0]  # 머리 2줄 + 이력 3줄
    assert ":red-background[✖ 미탐]" in table[3] and ":orange-background[✖ 오경보]" in table[4] and ":red-background" not in table[2]
    assert ":red[**NG**]" in table[2] and "| ✔ |" in table[2]
    assert "측정 안 함" in texts(at.caption)  # 화면을 처음 열 때 이미 있던 판정은 표시 지연을 재지 않는다
    assert fake.calls == ["status", "predictions", "image"]  # 처음 그릴 때: 상태 1 + 이력 1 + 이미지 1 (검사 상세는 읽지 않는다)
    assert fake.args["predictions"] == [{"limit": 20}]  # 이력 행 수 상한


def test_ok_latest_is_green_misjudgement_is_spelled_out_and_slow_latency_is_error(fake):
    fake.world["items"] = [prediction(2, "OK", "NG"), prediction(1, "OK", "OK")]
    at = new_page().run()
    body = texts(at.markdown)
    assert "# :green[OK]" in body and "불일치 ✖ (미탐: 불량을 OK로 판정)" in body
    fake.world["items"] = [prediction(3, "NG", "OK", ms=5200)]
    at.run()
    body = texts(at.markdown)
    assert "불일치 ✖ (오경보: 정상을 NG로 판정)" in body and "서버 처리 5200 ms (기준 5000 ms 초과 ✖)" in body


def test_start_uses_status_not_response_and_disables_buttons(fake):
    fake.world["status"] = stopped()
    at = new_page().run()
    at.run()  # 버튼 활성은 '직전에 읽은 상태' 기준이라 첫 갱신 다음부터 반영된다
    assert at.button(key="inspection_stop").disabled and not at.button(key="inspection_start").disabled
    assert "정지 · 드리프트: 꺼짐" in texts(at.markdown)
    click(at, "inspection_start")
    assert fake.args["start"] == [{"defect_rate": 0.2, "interval_seconds": 2.0}]
    assert fake.calls.index("start") < len(fake.calls) - 1 and fake.calls[fake.calls.index("start") + 1] == "status"  # 누른 뒤 상태를 다시 읽는다
    assert "동작중 · 간격 2.0초 · 불량률 20%" in texts(at.markdown)  # 응답 본문({"running": False})이 아니라 status 값
    assert "시작 요청을 보냈습니다" in texts(at.success)
    at.run()  # 버튼 활성은 직전에 읽은 상태 기준이라, 누른 직후 한 번(최대 4초) 늦게 바뀐다
    assert at.button(key="inspection_start").disabled and not at.button(key="inspection_stop").disabled
    assert at.number_input(key=inspection.RATE_KEY).disabled


def test_start_409_is_info_and_stop_is_idempotent(fake):
    fake.world["status"] = stopped()
    at = new_page().run()
    fake.fail["start"] = api_client.ApiError(409, "conflict", "이미 실행 중입니다")
    fake.world["status"] = running()  # 다른 곳에서 먼저 켠 경우
    click(at, "inspection_start")
    assert "이미 실행 중입니다. 현재 상태를 다시 확인했습니다." in texts(at.info) and not at.error
    assert "동작중" in texts(at.markdown)
    at.run()  # 방금 읽은 상태(동작중)가 버튼에 반영된다
    fake.world["status"] = stopped()  # 이미 다른 곳에서 멈춘 상태에서 중지를 누른다
    click(at, "inspection_stop")
    assert fake.calls.count("stop") == 1 and "자동 생성이 멈춘 상태입니다" in texts(at.success) and not at.error
    assert "정지 · 드리프트: 꺼짐" in texts(at.markdown)


def test_drift_on_off_bodies_and_status(fake):
    at = new_page().run()
    at.selectbox(key=inspection.DRIFT_KEY).select("noise")
    click(at, "inspection_drift_on")
    assert fake.args["drift"][-1] == {"enabled": True, "drift_type": "noise"}
    assert "드리프트 주입: 노이즈" in texts(at.markdown)
    click(at, "inspection_drift_off")
    assert fake.args["drift"][-1] == {"enabled": False, "drift_type": None}  # 끌 때는 종류를 보내지 않는다
    assert "드리프트: 꺼짐" in texts(at.markdown)


def test_manual_generate_predict_and_elapsed_text(fake):
    at = new_page().run()
    fake.calls.clear()
    click(at, "inspection_manual")
    assert fake.calls[:3] == ["generate", "predict", "image"] and fake.args["generate"] == [{"scenario": "normal"}]  # 정상만
    assert "이번 수동 판정: 생성→판정→이미지 받기까지 1.3초 (5초 이내 ✔, 브라우저 그리는 시간 제외)" in texts(at.success)
    assert len(fake.calls) <= 6 and "inspection" not in fake.calls  # 버튼이 낀 실행의 호출 상한, 검사 상세는 읽지 않는다
    assert fake.calls.count("image") == 1  # 수동으로 받은 이미지를 캐시에 넣어서, 같은 검사를 다시 받지 않는다
    assert "검사 INS-000004" in texts(at.markdown)
    click(at, "inspection_manual")  # 두 번째 클릭은 5.6초 걸린 것으로 한다
    assert "5.6초 (5초 초과 ✖" in texts(at.error)


def test_manual_predict_503_says_inspection_exists_but_not_judged(fake):
    at = new_page().run()
    fake.fail["predict"] = api_client.ApiError(503, "service_unavailable", "운영 중인 모델이 없습니다")
    click(at, "inspection_manual")
    assert "검사 INS-000004는 만들었지만 판정하지 못했습니다 (503: 운영 중인 모델이 없습니다)" in texts(at.error)


def test_manual_generate_failure_and_403(fake):
    at = new_page().run()
    fake.fail["generate"] = api_client.ApiError(400, "bad_request", "잘못된 요청")
    click(at, "inspection_manual")
    assert "1건 생성에 실패했습니다 (400: 잘못된 요청)" in texts(at.error)
    fake.fail["stop"] = api_client.ApiError(403, "forbidden", "권한 없음")
    click(at, "inspection_stop")
    assert "이 계정에는 권한이 없습니다." in texts(at.error)


def test_401_logs_out_once(fake):
    fake.fail["predictions"] = api_client.ApiError(401, "unauthorized", "만료")
    at = new_page().run()
    assert not at.exception and state.AUTH_KEY not in at.session_state
    assert at.session_state[state.LOGIN_ERROR_KEY] == inspection.SESSION_EXPIRED
    for key in state.INSPECTION_KEYS:
        assert key not in at.session_state
    fake.fail.clear()  # 다시 로그인한 뒤, 버튼 처리 중 401 도 같은 방식으로 로그아웃한다
    at.session_state[state.AUTH_KEY] = {"token": "tok", "role": "operator", "username": "operator", "expires_at": time.time() + 3600}
    at.run()
    assert state.AUTH_KEY in at.session_state
    fake.fail["stop"] = api_client.ApiError(401, "unauthorized", "만료")
    click(at, "inspection_stop")
    assert state.AUTH_KEY not in at.session_state


def test_server_down_keeps_last_values_and_skips_remaining_calls(fake):
    at = new_page().run()
    fake.fail["status"] = api_client.ServerDown("refused")
    fake.calls.clear()
    at.run()
    assert not at.exception and "서버에 연결할 수 없습니다" in texts(at.error)
    assert "기준입니다" in texts(at.caption) and "INS-000003" in texts(at.markdown)
    assert fake.calls == ["status"]  # 첫 호출이 실패하면 나머지는 부르지 않는다


def test_server_down_before_any_success_and_on_button(fake):
    fake.fail["status"] = api_client.ServerDown("refused")
    at = new_page().run()
    assert "서버에 연결할 수 없습니다" in texts(at.error) and "아직 성공한 갱신이 없습니다" in texts(at.caption)
    fake.fail["stop"] = api_client.ServerDown("timeout")
    click(at, "inspection_stop")
    assert "처리 여부는 다음 갱신에서 확인됩니다" in texts(at.error)


def test_same_id_is_not_fetched_again_and_new_id_measures_delay(fake):
    at = new_page().run()
    fake.calls.clear()
    at.run()
    assert fake.calls == ["status", "predictions"]  # 평소(새 판정 없음)는 2번, 이미지를 다시 받지 않는다
    fake.world["items"] = [prediction(4)] + fake.world["items"]
    fake.calls.clear()
    at.run()
    assert fake.calls == ["status", "predictions", "inspection", "image"]  # 자동 갱신 최대 4번
    assert "약 3.4초" in texts(at.caption) and "참고값" in texts(at.caption)
    fake.calls.clear()
    at.run()
    assert fake.calls == ["status", "predictions"] and "약 3.4초" in texts(at.caption)  # 한 번 잰 값은 유지된다


def test_stale_cache_does_not_measure_delay(fake):
    at = new_page().run()
    cache = at.session_state[state.INSPECTION_CACHE_KEY]
    at.session_state[state.INSPECTION_CACHE_KEY] = {**cache, "checked_at": time.time() - 3600}  # 한참 다른 화면에 있다가 돌아온 경우
    fake.world["items"] = [prediction(4)] + fake.world["items"]
    fake.calls.clear()
    at.run()
    assert "inspection" not in fake.calls and "측정 안 함" in texts(at.caption)


def test_missing_image_is_cached_and_not_retried(fake):
    fake.fail["image"] = api_client.ApiError(404, "not_found", "이미지를 찾을 수 없습니다")
    at = new_page().run()
    assert "이미지를 불러올 수 없습니다" in texts(at.caption) and not at.get("image")
    fake.calls.clear()
    at.run()
    assert "image" not in fake.calls  # 같은 id 를 갱신마다 다시 시도하지 않는다
