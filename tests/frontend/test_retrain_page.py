"""frontend/pages/retrain.py 화면 시험: streamlit.testing 의 AppTest + 가짜 ApiClient (실제 서버·네트워크·data/·models/ 없음)."""
import time

import pytest

import frontend.api_client as api_client
from frontend import state, view
from frontend.pages import retrain
from tests.frontend.test_view_retrain import gate, metrics, run


def model(version, acc_drift):
    return {"version": version, "stage": "staging", "metrics": metrics(1.0, acc_drift)}


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

    def training_runs(self, limit=1):
        self._do("runs", limit=limit)
        return {"items": FakeApi.world["runs"][:limit], "total": len(FakeApi.world["runs"])}

    def inspections(self, since=None, data_split=None, limit=1):
        self._do("pool", data_split=data_split, limit=limit)
        return {"items": [], "total": FakeApi.world["pool"]}

    def approvals(self, status=None, limit=1):
        self._do("approvals", status=status, limit=limit)
        return {"items": FakeApi.world["approvals"], "total": len(FakeApi.world["approvals"])}

    def model(self, version):
        self._do("model", version=version)
        if version not in FakeApi.world["models"]:
            raise api_client.ApiError(404, "not_found", "없는 버전")
        return FakeApi.world["models"][version]

    def start_training_run(self):
        self._do("start")
        FakeApi.world["runs"].insert(0, run("queued", None, n=len(FakeApi.world["runs"]) + 1, started=None, finished=None))
        return {"status": "succeeded"}  # 일부러 틀린 본문: 화면은 응답이 아니라 GET /training/runs 를 믿어야 한다


@pytest.fixture
def fake(monkeypatch):
    FakeApi.world = {"runs": [], "pool": 5, "approvals": [], "models": {"v1": model("v1", 0.854), "v2": model("v2", 0.97)}}
    FakeApi.calls, FakeApi.args, FakeApi.fail = [], {}, {}
    monkeypatch.setattr(api_client, "ApiClient", FakeApi)
    return FakeApi


def entry():
    from frontend.pages import retrain as page

    page.render()


def new_page(logged_in=True):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(entry, default_timeout=15)
    if logged_in:
        at.session_state[state.AUTH_KEY] = {"token": "tok", "role": "operator", "username": "operator", "expires_at": time.time() + 3600}
    return at


def texts(elements):
    return " | ".join(str(e.value) for e in elements)


def click(at):
    at.button(key=retrain.BUTTON_KEY).click().run()
    assert not at.exception


def approval(status):
    return {"approval_id": "APR-000002", "model_version": "v2", "requested_by": "system", "status": status}


def test_logged_out_makes_no_calls_and_first_draw_without_history(fake):
    at = new_page(logged_in=False).run()
    assert not at.exception and "로그인해 주세요" in texts(at.info) and fake.calls == []
    at = new_page().run()
    assert not at.exception and fake.calls == ["runs", "pool"]
    assert fake.args["runs"] == [{"limit": 10}] and fake.args["pool"] == [{"data_split": "pool", "limit": 1}]
    body = texts(at.markdown)
    assert ":green[5건]" in body and "학습 이력 없음" in body and not at.button(key=retrain.BUTTON_KEY).disabled
    assert "아직 재학습을 한 적이 없습니다" in texts(at.caption)
    captions = texts(at.caption)  # 한계 문구 3줄
    assert "한계 ① pool이 적으면" in captions and "30/90회" in captions and "한계 ② 평가(evaluate)" in captions and "안내 ③ 게이트를 통과하면" in captions
    assert "학습 중 새로 쌓인 검사" not in captions and len(at.exception) == 0  # 끝난 실행이 없으면 pool 편입 설명도 없다


def test_progress_comes_only_from_server_and_is_restored_in_a_new_session(fake):
    fake.world["runs"] = [run("running", "train", gate_result=None, finished=None, n=1)]
    at = new_page().run()
    body = texts(at.markdown)
    assert "실행 중: 학습" in body and ":green[✔ 완료]" in body and ":blue[진행 중]" in body and body.count(":gray[대기]") == 2
    at.run()  # 버튼 잠금은 직전에 읽은 서버 상태 기준이라 다음 갱신부터 반영된다
    assert at.button(key=retrain.BUTTON_KEY).disabled and "진행 중이라 버튼을 잠갔습니다 (직전 확인 기준" in texts(at.caption)
    assert "결과" not in [m.value for m in at.markdown]  # 끝나지 않았으면 결과 영역은 없다
    fake.world["runs"][0] = run("running", "data", gate_result=None, finished=None, n=1)  # 서버가 단계를 바꾸면 그대로 따라간다
    assert "실행 중: 데이터 구성" in texts(new_page().run().markdown)


def test_start_button_posts_once_then_rereads_server_state(fake):
    at = new_page().run()
    fake.calls.clear()
    click(at)
    assert fake.calls == ["start", "runs", "pool"] and fake.args["start"] == [{}]
    assert "재학습 요청을 보냈습니다" in texts(at.success)
    assert "대기 중 (곧 시작합니다)" in texts(at.markdown)  # 응답 본문({"status": "succeeded"})이 아니라 서버의 queued
    at.run()
    assert at.button(key=retrain.BUTTON_KEY).disabled


def test_409_busy_and_409_empty_pool_are_told_apart_by_server_state(fake):
    fake.world["pool"] = 0
    at = new_page().run()
    fake.fail["start"] = api_client.ApiError(409, "conflict", "아무 말")
    click(at)
    assert "새 데이터(pool)가 0건입니다" in texts(at.warning) and "이미 재학습이 실행 중" not in texts(at.info)
    fake.world["pool"] = 7  # 다른 곳에서 먼저 시작한 경우: 직전 화면은 한가했지만 서버는 실행 중
    fake.world["runs"] = [run("running", "train", gate_result=None, finished=None, n=1)]
    click(at)
    assert "이미 재학습이 실행 중입니다" in texts(at.info)
    fake.world["runs"] = []
    at.run().run()  # 버튼 잠금은 직전에 읽은 서버 상태 기준이라, 풀리는 데 갱신이 두 번 필요하다
    click(at)
    assert "받아들여지지 않았습니다 (409): 아무 말" in texts(at.warning)  # 둘 다 아니면 서버 메시지 그대로


def test_start_errors_403_503_server_down_and_401(fake):
    at = new_page().run()
    for error, expected in ((api_client.ApiError(403, "forbidden", "x"), "이 계정에는 재학습 권한이 없습니다."),
                            (api_client.ApiError(503, "service_unavailable", "고정 데이터셋이 없습니다"), "고정 데이터셋이 없어 시작하지 못했습니다")):
        fake.fail["start"] = error
        click(at)
        assert expected in texts(at.error)
    fake.fail["start"] = api_client.ServerDown("down")
    click(at)
    assert "처리 여부는 다음 갱신에서 확인됩니다" in texts(at.warning) and "서버에 연결할 수 없습니다" not in texts(at.error)  # 읽기는 되는 상태
    fake.fail["start"] = api_client.ApiError(401, "unauthorized", "만료")
    click(at)
    assert state.AUTH_KEY not in at.session_state and at.session_state[state.LOGIN_ERROR_KEY] == retrain.SESSION_EXPIRED
    for key in state.RETRAIN_KEYS:
        assert key not in at.session_state


def test_server_down_keeps_last_values_and_skips_remaining_calls(fake):
    fake.world["runs"] = [run("failed", "train", error="x", n=1)]
    at = new_page().run()
    fake.fail["runs"] = api_client.ServerDown("down")
    fake.calls.clear()
    at.run()
    assert not at.exception and fake.calls == ["runs"]  # 첫 호출에서 멈추고 나머지는 건너뛴다
    assert "서버에 연결할 수 없습니다" in texts(at.error) and "기준입니다 (서버가 꺼져 있어" in texts(at.caption)
    assert "RUN-000001" in texts(at.markdown) and ":gray[" in texts(at.markdown)  # 마지막 성공 값을 흐리게
    fake.fail.clear()
    fake.fail["runs"] = api_client.ApiError(401, "unauthorized", "만료")
    at.run()
    assert state.AUTH_KEY not in at.session_state


def test_failed_run_shows_raw_error_in_code_and_escaped_in_table(fake):
    fake.world["runs"] = [run("failed", "train", error="디스크|오류\n:red[x] $1", n=1)]
    at = new_page().run()
    assert not at.exception and fake.calls == ["runs", "pool"]  # 실패는 게이트·승인·모델 상세를 읽지 않는다
    assert "재학습 실패" in texts(at.error) and "통과하지 못했습니다" not in texts(at.warning)
    assert [c.value for c in at.code] == ["디스크|오류\n:red[x] $1"]
    assert "✖ 실패" in texts(at.markdown) and "편입되지 않았습니다" in texts(at.caption)
    table = [m.value for m in at.markdown if str(m.value).startswith("| 실행")][0].splitlines()
    assert len(table) == 3 and "\\|" in table[2] and "\\[x\\]" in table[2] and "\\$1" in table[2] and "\n" not in table[2]


def test_gate_failed_run_is_not_a_failed_run_and_has_no_approval_lookup(fake):
    bad = gate(False)
    fake.world["runs"] = [run(gate_result=bad, n=1)]
    at = new_page().run()
    assert not at.exception and fake.calls == ["runs", "pool", "model", "model"] and fake.args["model"] == [{"version": "v2"}, {"version": "v1"}]
    assert "통과하지 못했습니다" in texts(at.warning) and "승인 요청은 만들어지지 않았고" in texts(at.warning) and not at.error
    body = texts(at.markdown)
    assert ":red[✖ 미통과]" in body and ":green[✔ 통과]" in body and "G3 미통과" in body and "게이트가 비교한 기준 모델: v1" in body
    assert "pool 120장은 train으로 편입되어 pool 건수가 줄었습니다" in texts(at.caption)
    assert "학습 중 새로 쌓인 검사는 pool에 남습니다" in texts(at.caption) and "train 2000장 (고정 데이터 포함)" in texts(at.caption)
    assert "+11.6" in body and "TN 400 · FP 0 · FN 0 · TP 100" in body and "기존 = v1, 후보 = v2" in texts(at.caption)
    assert "성공 (게이트 미통과)" in body


def test_gate_passed_shows_real_approval_status_and_caches_models(fake):
    fake.world["runs"] = [run(n=1)]
    fake.world["approvals"] = [approval("pending")]
    at = new_page().run()
    assert fake.calls == ["runs", "pool", "approvals", "model", "model"] and len(fake.calls) <= 5
    assert fake.args["approvals"] == [{"status": None, "limit": 20}]  # 상태 필터 없이 읽는다
    assert "승인 요청 APR-000002 (v2): 승인 대기" in texts(at.info)
    fake.calls.clear()
    fake.world["approvals"] = [approval("approved")]  # 샌드박스에서 승인한 경우: 같은 요청의 실제 status 로 바뀐다
    at.run()
    assert fake.calls == ["runs", "pool", "approvals"] and "승인 요청 APR-000002 (v2): 승인됨" in texts(at.success)  # 모델 상세는 다시 받지 않는다
    assert "승인 대기" not in texts(at.success) + texts(at.info)
    fake.world["approvals"] = [approval("rejected")]
    assert "거절됨" in texts(at.run().warning)
    fake.world["approvals"] = []
    assert "최근 20건에서 찾지 못했습니다" in texts(at.run().warning)
    fake.world["pool"] = 0  # 편입으로 pool 이 줄어든 모습
    assert ":gray[0건]" in texts(at.run().markdown)


def test_missing_baseline_and_failed_model_read_keep_gate_rows(fake):
    fake.world["runs"] = [run(n=1)]
    fake.world["approvals"] = [approval("pending")]
    fake.world["models"].pop("v1")
    at = new_page().run()
    assert not at.exception and "기준 모델 v1의 지표를 읽지 못했습니다" in texts(at.caption)
    assert ":green[✔ 통과]" in texts(at.markdown) and "없음" in texts(at.markdown)  # 게이트 행은 그대로, 기존 값은 '없음'
    fake.world["models"].pop("v2")
    fake.world["runs"] = [run(n=2, gate_result=gate(True, None, []))]  # 게이트 미적용
    at = new_page().run()
    assert "게이트 미적용" in texts(at.info) and "게이트 미적용 (첫 모델)" in texts(at.markdown) and "후보 모델의 지표를 읽지 못했습니다" in texts(at.caption)


def test_history_has_at_most_ten_rows_and_401_in_snapshot_logs_out(fake):
    fake.world["runs"] = [run(n=n, gate_result=gate(n % 2 == 0)) for n in range(12, 0, -1)]
    at = new_page().run()
    table = [m.value for m in at.markdown if str(m.value).startswith("| 실행")][0].splitlines()
    assert len(table) == 2 + 10 and "전체 12건 중 최근 10건" in texts(at.caption) and "RUN-000012" in table[2] and "RUN-000003" in table[11]
    fake.fail["pool"] = api_client.ApiError(401, "unauthorized", "만료")
    at.run()
    assert state.AUTH_KEY not in at.session_state


def test_view_and_page_use_same_refresh_note():
    from frontend import config

    assert config.refresh_seconds() == 4 and retrain.view is view
