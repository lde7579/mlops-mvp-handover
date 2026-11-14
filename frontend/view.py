"""화면에 보일 문장·서식을 만드는 순수 함수 (Streamlit 을 쓰지 않아서 빠르게 시험할 수 있다).

수준(level)은 "ok"(정상), "warn"(주의), "error"(오류) 세 가지이고, 화면이 색으로 바꿔서 보여 준다.
"""
from typing import Optional

ROLE_LABELS = {"operator": "운영자", "admin": "관리자"}


def role_label(role: Optional[str]) -> str:
    return ROLE_LABELS.get(role or "", role or "알 수 없음")


def backend_status(health: Optional[dict]) -> tuple:
    """사이드바 '백엔드 상태'. health 가 None 이면 서버에 연결하지 못한 것. 반환: (문장, 수준)."""
    if health is None:
        return "연결 안 됨", "error"
    if health.get("database") != "ok":
        return "DB 오류 (서버는 응답함)", "error"
    if health.get("status") != "ok":
        return "일부 문제 있음", "warn"
    return "정상", "ok"


def model_status(health: Optional[dict]) -> tuple:
    """사이드바 '운영 모델'. 반환: (문장, 수준)."""
    if health is None:
        return "확인 불가", "error"
    model = health.get("model") or {}
    version = model.get("production_version")
    if not version:
        return "운영 모델 없음", "warn"
    if model.get("loaded"):
        return f"{version} (불러옴)", "ok"
    return f"{version} (아직 불러오지 못함)", "warn"


def login_error_message(status: Optional[int], server_down: bool, url: str) -> str:
    """로그인 실패 문장. 서버가 꺼졌으면 server_down=True, 서버 오류 응답이면 status 에 상태 코드."""
    if server_down:
        return f"서버에 연결할 수 없습니다 ({url}). 서버를 켠 뒤 다시 시도해 주세요."
    if status == 401:
        return "계정 또는 비밀번호가 틀렸습니다."
    if status == 400:
        return "입력한 내용을 확인해 주세요."
    return f"로그인하지 못했습니다 (서버 응답 {status})."


def session_text(role: Optional[str], username: Optional[str]) -> str:
    return f"{username} ({role_label(role)})"


def is_expired(expires_at: Optional[float], now: float) -> bool:
    """토큰 만료 시각(초 단위 시각)이 지났거나 없으면 만료로 본다."""
    return expires_at is None or now >= expires_at


# ---------- 대시보드 (④-1b) ----------
# 조회 결과 한 건은 {"ok", "data", "error", "status"} 모양이다. 화면 코드는 이 모양만 알면 되고, 문장·서식은 아래 함수들이 만든다.
# 수준(level): "ok"(정상) "warn"(주의) "error"(오류) "info"(진행 중) "muted"(해당 없음·보류)

DRIFT_LABELS = {"dark": "어두움", "noise": "노이즈", "shift": "이동"}
STEP_LABELS = {"data": "데이터 구성", "train": "학습", "evaluate": "평가", "gate": "게이트"}


def ok_result(data) -> dict:
    return {"ok": True, "data": data, "error": None, "status": None}


def fail_result(status: Optional[int], message: str) -> dict:
    return {"ok": False, "data": None, "error": message, "status": status}


def today_start(now) -> str:
    """오늘 0시 (시간대 포함 ISO 문자열). now 는 시간대가 있는 datetime. API 는 시간대 없는 값을 400 으로 거절한다."""
    return now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


def clock_text(ts: float) -> str:
    import time

    return time.strftime("%H:%M:%S", time.localtime(ts))


def pct(value: Optional[float]) -> str:
    return "없음" if value is None else f"{value * 100:.1f}%"


def server_down_text(url: str) -> str:
    return f"서버에 연결할 수 없습니다 ({url}). 서버를 켠 뒤 기다리면 자동으로 다시 연결합니다."


def last_success_text(ts: Optional[float], seconds: int) -> str:
    if ts is None:
        return f"아직 성공한 갱신이 없습니다 ({seconds}초마다 자동 갱신)"
    return f"마지막 갱신 성공: {clock_text(ts)} ({seconds}초마다 자동 갱신)"


def _card(label: str, value: str, level: str = "ok", caption: str = "") -> dict:
    return {"label": label, "value": value, "level": level, "caption": caption}


def _card_failed(label: str, res: dict) -> dict:
    return _card(label, "조회 실패", "error", res.get("error") or "")


def card_production(res: dict) -> dict:
    if not res["ok"]:
        return _card_failed("운영 모델", res)
    if res["data"] is None:  # 404: 운영 모델이 아직 없다 (정상적인 상태)
        return _card("운영 모델", "운영 모델 없음", "warn")
    return _card("운영 모델", res["data"]["version"])


def _total_card(label: str, res: dict, unit: str, level_if_positive: str = "ok") -> dict:
    if not res["ok"]:
        return _card_failed(label, res)
    total = res["data"]["total"]
    return _card(label, f"{total}{unit}", level_if_positive if total > 0 else "ok")


def card_today(res: dict) -> dict:
    return _total_card("오늘 검사 건수", res, "건")


def card_pool(res: dict) -> dict:
    return _total_card("pool 건수", res, "건")


def card_pending(res: dict) -> dict:
    return _total_card("승인 대기", res, "건", "warn")


def card_ng_rate(res: dict) -> dict:
    label = "불량률 (판정 기준, 전체)"
    if not res["ok"]:
        return _card_failed(label, res)
    data = res["data"]
    if data["ng_rate"] is None:
        return _card(label, "판정 없음", "muted")
    return _card(label, pct(data["ng_rate"]), "ok", f"판정 {data['total']}건 중 NG {data['ng_count']}건 (전체 누적)")


def card_accuracy(res: dict) -> dict:
    label = "최근 정확도 · 재현율"
    if not res["ok"]:
        return _card_failed(label, res)
    recent, size = res["data"]["recent"], res["data"]["criteria"]["recent_n"]
    n, accuracy = recent["n"], recent["accuracy"]
    parts = [f"최근 {n}/{size}건" if n < size else f"최근 {n}건"]
    if accuracy is None:
        return _card(label, "판정 없음", "muted", parts[0])
    parts.append("재현율 " + (pct(recent["recall"]) if recent["recall"] is not None else "해당 없음(정답 NG 없음)"))
    return _card(label, pct(accuracy), "ok", " · ".join(parts))


def cards(snap: dict) -> list:
    return [card_production(snap["production"]), card_today(snap["today"]), card_ng_rate(snap["stats"]),
            card_accuracy(snap["performance"]), card_pending(snap["approvals"]), card_pool(snap["pool"])]


def needs_deployment_check(drift_res: dict) -> bool:
    """판단 보류(active 가 null)이고 판정이 N건 미만일 때만 최신 배포 기록이 필요하다 (불필요한 호출을 줄인다)."""
    if not drift_res["ok"]:
        return False
    warning = drift_res["data"]["performance_warning"]
    return warning["active"] is None and warning["model_version"] is not None and warning["n"] < warning["recent_n"]


def performance_warning(drift_res: dict, deployments_res: Optional[dict]) -> tuple:
    """성능 경고 배너 (문장, 수준). 수준: error(빨강) / ok(초록) / muted(회색 판단 보류)."""
    if not drift_res["ok"]:
        return "경고 상태를 확인할 수 없습니다. " + (drift_res.get("error") or ""), "muted"
    w = drift_res["data"]["performance_warning"]
    size, n = w["recent_n"], w["n"]
    basis = f"기준 {pct(w['accuracy_threshold'])}" + ("" if w["confirmed"] else " (권장값)")
    if w["active"] is True:
        return (f"성능 저하 경고: 최근 {n}건 정확도 {pct(w['accuracy'])} < {basis}. "
                "재학습 화면에서 재학습을 실행할 수 있습니다."), "error"
    if w["active"] is False:
        return f"정상: 최근 {n}건 정확도 {pct(w['accuracy'])} ({basis} 이상)", "ok"
    version = w["model_version"]
    if version is None:
        return "판단 보류: 운영 모델이 없어 판단을 보류합니다.", "muted"
    latest = None
    if deployments_res is not None and deployments_res["ok"] and deployments_res["data"]["items"]:
        latest = deployments_res["data"]["items"][0]
    if n < size and latest is not None and latest["version"] == version:
        return (f"판단 보류: 방금 모델이 {version}으로 바뀌었습니다. 성능 경고는 새 모델이 판정한 최근 {size}건으로만 판단하므로, "
                f"새 판정이 {size}건 쌓일 때까지(2초 간격이면 약 60초) 판단을 보류합니다. "
                "이전 모델의 판정 때문에 경고가 남지 않게 하려는 규칙입니다."), "muted"
    return f"판단 보류: 판정이 {n}/{size}건이라 판단을 보류합니다.", "muted"


def stage_collect(res: dict) -> tuple:
    if not res["ok"]:
        return "확인 불가", "error"
    s = res["data"]
    if s["running"]:
        text, level = "동작중", "ok"
    else:
        text, level = "정지", "muted"
        if s.get("last_error"):
            text, level = f"스스로 멈춤: {s['last_error']}", "error"
    drift = s["drift"]
    if drift["enabled"]:
        text += f" · 드리프트 주입 중: {DRIFT_LABELS.get(drift['drift_type'], drift['drift_type'])}"
        if level != "error":
            level = "warn"
    return text, level


def stage_infer(res: dict) -> tuple:
    return ("확인 불가", "error") if not res["ok"] else model_status(res["data"])


def stage_train(res: dict) -> tuple:
    if not res["ok"]:
        return "확인 불가", "error"
    items = res["data"]["items"]
    if not items:
        return "대기 (학습 이력 없음)", "muted"
    run = items[0]
    if run["status"] == "queued":
        return "실행 대기 중", "info"
    if run["status"] == "running":
        return f"실행 중: {STEP_LABELS.get(run['step'], run['step'] or '준비')}", "info"
    if run["status"] == "failed":
        return "실패", "error"
    if (run.get("gate_result") or {}).get("passed") is False:
        return f"성공 ({run['result_version']}), 게이트 미통과", "warn"
    return f"성공 ({run['result_version']})", "ok"


def stage_deploy(res: dict) -> tuple:
    if not res["ok"]:
        return "확인 불가", "error"
    return ("운영 모델 없음", "warn") if res["data"] is None else (res["data"]["version"], "ok")


def stages(snap: dict) -> list:
    """맨 위 단계 줄: [(이름, 문장, 수준)] 수집 → 추론 → 학습 → 배포."""
    return [("수집", *stage_collect(snap["simulator"])), ("추론", *stage_infer(snap["health"])),
            ("학습", *stage_train(snap["training"])), ("배포", *stage_deploy(snap["production"]))]


def has_unauthorized(snap: dict) -> bool:
    return any(r is not None and r["status"] == 401 for r in snap.values())


# ---------- 검사/추론 화면 (④-2) ----------
# 양성은 NG (/predictions/stats 의 혼동행렬과 같은 정의): 정답 NG 인데 OK 로 판정 = 미탐(불량을 놓침), 정답 OK 인데 NG 로 판정 = 오경보.
LATENCY_LIMIT_MS = 5000  # e2e 2단계 합격 기준: 모든 latency_ms <= 5000
ELAPSED_LIMIT_SECONDS = 5.0  # e2e 2단계 서술: 5초 이내 OK/NG 표시
REFRESH_NOTE = "4초"  # config.REFRESH_SECONDS 와 같은 값이어야 한다 (문구용, 시험이 확인한다)
ROW_STYLES = {"missed": "background-color: #ffd6d6; color: #000000", "false_alarm": "background-color: #ffe6c7; color: #000000"}


def misjudgement_kind(true_label: str, predicted_label: str) -> Optional[str]:
    """None(일치) / "missed"(미탐) / "false_alarm"(오경보)."""
    if true_label == predicted_label:
        return None
    return "missed" if true_label == "NG" else "false_alarm"


def match_text(kind: Optional[str]) -> str:
    return {"missed": "✖ 미탐", "false_alarm": "✖ 오경보"}.get(kind, "✔")


def match_detail(true_label: str, kind: Optional[str]) -> str:
    """최신 카드의 정답 대조 줄."""
    if kind is None:
        return f"정답: {true_label} — 일치 ✔"
    if kind == "missed":
        return f"정답: {true_label} — 불일치 ✖ (미탐: 불량을 OK로 판정)"
    return f"정답: {true_label} — 불일치 ✖ (오경보: 정상을 NG로 판정)"


def row_style(kind: Optional[str]) -> str:
    """이력 표 행의 CSS. 정판은 빈 문자열 (글자 색은 밝은/어두운 테마 모두에서 읽히도록 배경과 함께 정한다)."""
    return ROW_STYLES.get(kind, "")


def parse_utc(text: str):
    """API 의 UTC 'Z' 시각 문자열 -> 시간대가 있는 datetime."""
    from datetime import datetime

    return datetime.fromisoformat(text)


def seconds_between(earlier_text: Optional[str], later) -> Optional[float]:
    """API 시각 문자열 earlier_text 부터 later(시간대 있는 datetime)까지의 초. 값이 없거나 읽을 수 없으면 None."""
    if not earlier_text:
        return None
    try:
        return (later - parse_utc(earlier_text)).total_seconds()
    except (ValueError, TypeError):
        return None


def local_time_text(utc_text: str) -> str:
    try:
        return parse_utc(utc_text).astimezone().strftime("%H:%M:%S")
    except (ValueError, TypeError):
        return utc_text


def history_rows(items: list) -> list:
    """GET /predictions 의 items -> 표 행. '_kind' 는 행 강조용이라 화면에는 보이지 않게 쓴다."""
    rows = []
    for item in items:
        kind = misjudgement_kind(item["true_label"], item["predicted_label"])
        rows.append({"시각": local_time_text(item["predicted_at"]), "검사번호": item["inspection_id"], "모델": item["model_version"],
                     "판정": item["predicted_label"], "점수": f"{item['score']:.2f}", "정답": item["true_label"],
                     "정오": match_text(kind), "처리(ms)": item["latency_ms"], "_kind": kind})
    return rows


HISTORY_COLUMNS = ("시각", "검사번호", "모델", "판정", "점수", "정답", "정오", "처리(ms)")
ROW_DIRECTIVES = {"missed": "red-background", "false_alarm": "orange-background"}  # st.markdown 의 색 지시어 (오판 행 강조)


def history_markdown(rows: list) -> str:
    """history_rows 결과 -> 마크다운 표. 오판 행은 모든 칸에 배경색(미탐 빨강, 오경보 주황), NG 판정은 빨간 굵은 글자,
    처리 시간이 5000 ms 를 넘으면 빨간 글자. (pandas 없이 st.markdown 으로 그린다.)"""
    lines = ["| " + " | ".join(HISTORY_COLUMNS) + " |", "|" + "---|" * len(HISTORY_COLUMNS)]
    for row in rows:
        cells = {name: str(row[name]) for name in HISTORY_COLUMNS}
        directive = ROW_DIRECTIVES.get(row["_kind"])
        if directive:  # 색 지시어는 겹쳐 쓰지 않는다: 오판 행은 배경색만
            cells = {name: f":{directive}[{text}]" for name, text in cells.items()}
        else:
            if row["판정"] == "NG":
                cells["판정"] = f":red[**{cells['판정']}**]"
            if row["처리(ms)"] > LATENCY_LIMIT_MS:
                cells["처리(ms)"] = f":red[{cells['처리(ms)']}]"
        lines.append("| " + " | ".join(cells[name] for name in HISTORY_COLUMNS) + " |")
    return "\n".join(lines)


def ng_summary_text(items: list) -> str:
    if not items:
        return "아직 판정이 없습니다."
    kinds = [misjudgement_kind(i["true_label"], i["predicted_label"]) for i in items]
    ng = sum(1 for i in items if i["predicted_label"] == "NG")
    return (f"최근 {len(items)}건 중 NG 판정 {ng}건, 오판 {sum(k is not None for k in kinds)}건 "
            f"(미탐 {kinds.count('missed')}, 오경보 {kinds.count('false_alarm')})")


def latency_text(ms: int) -> tuple:
    """(가) 서버 처리 시간. 반환: (문장, 수준)."""
    if ms <= LATENCY_LIMIT_MS:
        return f"서버 처리 {ms} ms (기준 {LATENCY_LIMIT_MS} ms 이내 ✔)", "ok"
    return f"서버 처리 {ms} ms (기준 {LATENCY_LIMIT_MS} ms 초과 ✖)", "error"


def elapsed_text(seconds: float) -> tuple:
    """(나) 수동 생성→판정→이미지 받기까지 경과. 반환: (문장, 수준)."""
    head = f"이번 수동 판정: 생성→판정→이미지 받기까지 {seconds:.1f}초"
    if seconds <= ELAPSED_LIMIT_SECONDS:
        return f"{head} ({ELAPSED_LIMIT_SECONDS:.0f}초 이내 ✔, 브라우저 그리는 시간 제외)", "ok"
    return f"{head} ({ELAPSED_LIMIT_SECONDS:.0f}초 초과 ✖, 브라우저 그리는 시간 제외)", "error"


def display_delay_text(seconds: Optional[float]) -> str:
    """(다) 자동 생성 표시 지연. 참고값이라 합격/불합격을 표시하지 않는다."""
    if seconds is None or seconds < 0:
        return "자동 생성 표시 지연: 측정 안 함 (화면을 열 때 이미 있던 판정이거나 시각을 비교할 수 없음)"
    return (f"자동 생성: 만들어진 뒤 화면에 나타나기까지 약 {seconds:.1f}초 "
            f"(참고값, {REFRESH_NOTE}마다 갱신하므로 {ELAPSED_LIMIT_SECONDS:.0f}초를 넘을 수 있음)")


def simulator_detail_text(status: dict) -> tuple:
    """GET /simulator/status 로 만든 상태 줄. 반환: (문장, 수준). 화면은 이 값만 믿는다."""
    _, level = stage_collect(ok_result(status))
    if status["running"]:
        parts = ["동작중", f"간격 {status['interval_seconds']:.1f}초", f"불량률 {status['defect_rate'] * 100:.0f}%"]
    else:
        parts = ["정지"]
    drift = status["drift"]
    parts.append("드리프트 주입: " + DRIFT_LABELS.get(drift["drift_type"], str(drift["drift_type"])) if drift["enabled"] else "드리프트: 꺼짐")
    parts.append(f"만든 {status['generated']}건 · 판정 {status['judged']}건 · 건너뜀 {status['judge_failures']}건")
    if status.get("last_error"):
        parts.append(f"마지막 오류: {status['last_error']}")
    return " · ".join(parts), level


ACTION_LABELS = {"start": "시작", "stop": "중지", "drift_on": "드리프트 켜기", "drift_off": "드리프트 끄기", "manual": "1건 생성+판정"}
SERVER_DOWN_ACTION_TEXT = "요청을 보내지 못했거나 응답을 받지 못했습니다. 처리 여부는 다음 갱신에서 확인됩니다."


def action_message(action: str, status: int, message: str) -> tuple:
    """버튼 요청이 오류 응답으로 끝났을 때의 문장. 반환: (문장, 수준). 401 은 화면이 따로 처리한다(로그아웃)."""
    if status == 409:
        return "이미 실행 중입니다. 현재 상태를 다시 확인했습니다.", "info"
    if status == 403:
        return "이 계정에는 권한이 없습니다.", "error"
    return f"{ACTION_LABELS.get(action, action)} 요청 실패 ({status}): {message}", "error"


def manual_failure_text(inspection_id: Optional[str], status: int, message: str) -> str:
    """수동 생성+판정 실패. 검사는 만들어졌는데 판정만 실패했으면 그 사실을 따로 알린다."""
    if inspection_id is not None:
        return f"검사 {inspection_id}는 만들었지만 판정하지 못했습니다 ({status}: {message})"
    return f"1건 생성에 실패했습니다 ({status}: {message})"


# ---------- 재학습 화면 (④-3) ----------
# 서버 값(오류 문장, 메시지 등)이 마크다운에 들어갈 때는 항상 md_cell 을 거친다 (표가 깨지거나 :red[ 같은 글자가 서식으로 읽히는 것을 막는다).
RUN_STATUS_LABELS = {"queued": "대기", "running": "실행", "succeeded": "성공", "failed": "실패"}
STEP_ORDER = ("data", "train", "evaluate", "gate")
APPROVAL_LABELS = {"pending": "승인 대기", "approved": "승인됨", "rejected": "거절됨"}
APPROVAL_LIMIT = 20  # 승인 요청을 찾으려고 읽는 최근 건수 (GET /approvals?limit=20, status 필터 없음)
HISTORY_LIMIT = 10
ERROR_PREVIEW_CHARS = 40
SERVER_DOWN_START_TEXT = "요청을 보냈지만 응답을 받지 못했습니다. 처리 여부는 다음 갱신에서 확인됩니다."
LIMIT_POOL_TEXT = ("pool이 적으면 회복이 불안정할 수 있습니다. 오프라인 측정(pool 30·60·100·150·200·300장, 90회)에서 G1~G3를 모두 통과한 것은 30/90회였습니다. "
                   "60장은 기준점이 불합격이어서, 리허설에서 통과를 확인하기 전에는 쓰지 않습니다.")
LIMIT_EVALUATE_TEXT = "평가(evaluate) 단계는 순간적으로 지나가서 화면에서 거의 보이지 않습니다 (실제 평가는 학습 단계 안에서 끝납니다)."
LIMIT_APPROVAL_TEXT = "게이트를 통과하면 승인 대기 요청이 자동으로 만들어집니다. 승인과 거절은 왼쪽 메뉴의 [승인] 화면에서 관리자 계정으로 합니다."
HISTORY_RUN_COLUMNS = ("실행", "상태", "시작", "소요", "결과 버전", "게이트", "학습 데이터", "오류")
GATE_COLUMNS = ("조건", "결과", "후보", "기존", "요구", "근거")
METRIC_COLUMNS = ("평가셋", "지표", "기존", "후보", "차이(%p)")
CONFUSION_COLUMNS = ("평가셋", "기존", "후보")
METRIC_NAMES = (("accuracy", "정확도"), ("recall", "재현율"), ("precision", "정밀도"), ("f1", "F1"))
EVAL_SETS = (("normal", "정상 (test_normal)"), ("drift", "드리프트 (test_drift)"))


def md_cell(text) -> str:
    """서버 값을 마크다운 안에서 글자 그대로 보이게 한다: \\ | ` [ ] $ * _ 앞에 \\ 를 붙이고 줄바꿈은 공백으로 바꾼다."""
    out = str(text).replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    out = out.replace("\\", "\\\\")
    for ch in "|`[]$*_":
        out = out.replace(ch, "\\" + ch)
    return out


def md_table(columns: tuple, rows: list, tint: Optional[dict] = None) -> str:
    """rows(칸 이름 -> 값) -> 마크다운 표. 모든 칸에 md_cell 을 적용한다. tint 는 {칸 이름: 행 -> 색 이름 또는 None}."""
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = []
        for name in columns:
            text = md_cell(row[name])
            color = (tint or {}).get(name, lambda r: None)(row)
            cells.append(f":{color}[{text}]" if color else text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _gate(run: Optional[dict]) -> Optional[dict]:
    return None if run is None else run.get("gate_result")


def is_run_active(run: Optional[dict]) -> bool:
    return run is not None and run["status"] in ("queued", "running")


def latest_run(runs_res: Optional[dict]) -> Optional[dict]:
    """GET /training/runs 조회 결과 -> 가장 최근 run (없거나 조회 실패면 None)."""
    if runs_res is None or not runs_res["ok"] or not runs_res["data"]["items"]:
        return None
    return runs_res["data"]["items"][0]


def run_stages(run: Optional[dict]) -> list:
    """진행 4칸 [(이름, 표시, 수준)]. 서버 run 의 status·step 만 본다."""
    names = [STEP_LABELS[s] for s in STEP_ORDER]
    waiting = [(n, "대기", "muted") for n in names]
    if run is None or run["status"] == "queued":
        return waiting
    if run["status"] == "succeeded":
        return [(n, "✔ 완료", "ok") for n in names]
    if run["step"] not in STEP_ORDER:  # 단계 기록 전에 끝난 실패 등: 칸 표시 없이 문장으로만 알린다
        return waiting
    at = STEP_ORDER.index(run["step"])
    current = ("진행 중", "info") if run["status"] == "running" else ("✖ 실패", "error")
    return [(n, "✔ 완료", "ok") if i < at else (n, *current) if i == at else (n, "대기", "muted") for i, n in enumerate(names)]


def run_status_text(run: Optional[dict]) -> tuple:
    if run is None:
        return "학습 이력 없음", "muted"
    if run["status"] == "queued":
        return "대기 중 (곧 시작합니다)", "info"
    if run["status"] == "running":
        return f"실행 중: {STEP_LABELS.get(run['step'], run['step'] or '준비')}", "info"
    if run["status"] == "failed":
        return "실패", "error"
    if (_gate(run) or {}).get("passed") is False:
        return "성공 (게이트 미통과)", "warn"
    return "성공", "ok"


def pool_text(pool_res: Optional[dict]) -> tuple:
    if pool_res is None or not pool_res["ok"]:
        return "pool 건수를 읽지 못했습니다" + ("" if pool_res is None else f" ({md_cell(pool_res['error'])})"), "error"
    total = pool_res["data"]["total"]
    return f"{total}건", "ok" if total > 0 else "muted"


def snapshot_text(run: Optional[dict]) -> str:
    snap = None if run is None else run.get("train_data_snapshot")
    if not snap:
        return "학습 데이터 요약: 없음 (결과 버전이 만들어진 실행에만 있습니다)"
    return f"학습 데이터 요약: train {snap.get('train')}장 (고정 데이터 포함) + 이번에 편입한 pool {snap.get('pool')}장"


def pool_note(run: Optional[dict]) -> Optional[str]:
    """실행이 끝난 뒤 pool 건수가 줄어드는 이유 (성공일 때) 또는 줄지 않는 이유 (실패일 때)."""
    if run is None:
        return None
    if run["status"] == "failed":
        return "실패했으므로 읽은 pool 데이터는 train으로 편입되지 않았습니다."
    snap = run.get("train_data_snapshot")
    if run["status"] == "succeeded" and snap:
        return (f"재학습에 쓴 pool {snap.get('pool')}장은 train으로 편입되어 pool 건수가 줄었습니다 "
                "(게이트 미통과여도 편입됩니다). 학습이 읽은 pool만 train으로 편입되고, 학습 중 새로 쌓인 검사는 pool에 남습니다.")
    return None


def needs_approval_check(run: Optional[dict]) -> bool:
    gate = _gate(run)
    return (run is not None and run["status"] == "succeeded" and bool(run.get("result_version"))
            and gate is not None and gate.get("passed") is True and gate.get("baseline_version") is not None)


def models_needed(run: Optional[dict]) -> list:
    """지표 비교에 필요한 모델 버전 [후보, 기준] (성공한 실행만, 없는 값은 뺀다)."""
    if run is None or run["status"] != "succeeded" or not run.get("result_version"):
        return []
    return [v for v in (run["result_version"], (_gate(run) or {}).get("baseline_version")) if v]


def approval_text(approvals_res: Optional[dict], version: str) -> tuple:
    """후보 버전의 승인 요청을 GET /approvals(필터 없음) 응답에서 찾아 실제 status 를 보여 준다."""
    if approvals_res is None or not approvals_res["ok"]:
        return "승인 요청을 확인하지 못했습니다" + ("" if approvals_res is None else f" ({md_cell(approvals_res['error'])})"), "warn"
    for item in approvals_res["data"]["items"]:
        if item["model_version"] == version:
            label = APPROVAL_LABELS.get(item["status"], md_cell(item["status"]))
            return f"승인 요청 {item['approval_id']} ({version}): {label}", {"pending": "info", "approved": "ok"}.get(item["status"], "warn")
    return f"승인 요청을 최근 {APPROVAL_LIMIT}건에서 찾지 못했습니다 ({version})", "warn"


def run_outcome(run: Optional[dict], approvals_res: Optional[dict]) -> Optional[tuple]:
    """결과 영역 맨 위 문장 (문장, 수준). 아직 끝나지 않았으면 None."""
    if run is None or run["status"] not in ("succeeded", "failed"):
        return None
    if run["status"] == "failed":
        return "재학습 실패: 새 모델이 만들어지지 않았고 운영 모델은 그대로입니다. 오류 내용은 아래에 그대로 표시합니다.", "error"
    gate, version = _gate(run), run.get("result_version")
    if gate is None:
        return f"재학습은 끝났지만 게이트 결과가 없습니다 ({version}).", "warn"
    if gate.get("baseline_version") is None:
        return f"게이트 미적용: 비교할 운영 모델이 없어 게이트를 적용하지 않았습니다 ({version}).", "info"
    if gate.get("passed") is False:
        return (f"재학습은 끝났지만 게이트를 통과하지 못했습니다 (후보 {version}는 rejected). 승인 요청은 만들어지지 않았고 운영 모델은 그대로입니다. "
                "어느 조건에서 떨어졌는지는 아래 G1~G3 행을 보세요."), "warn"
    text, level = approval_text(approvals_res, version)
    return f"재학습과 게이트 통과 ({version}). {text}", level


def gate_basis_text(run: Optional[dict]) -> str:
    gate = _gate(run)
    if gate is None:
        return "게이트 전 (아직 결과가 없습니다)"
    if gate.get("baseline_version") is None:
        return "게이트 미적용 (첫 모델)"
    return f"게이트가 비교한 기준 모델: {gate['baseline_version']}"


def gate_rows(gate: Optional[dict]) -> list:
    """gate_result.checks -> 표 행 (G1~G3). 게이트 전·미적용이면 빈 목록. '_passed' 는 색 지정용이라 표에는 나오지 않는다."""
    rows = []
    for c in (gate or {}).get("checks", []):
        rows.append({"조건": c["id"], "결과": "✔ 통과" if c["passed"] else "✖ 미통과", "후보": pct(c["candidate"]), "기존": pct(c["baseline"]),
                     "요구": c["required"], "근거": c["message"], "_passed": c["passed"]})
    return rows


def gate_markdown(rows: list) -> str:
    return md_table(GATE_COLUMNS, rows, tint={"결과": lambda r: "green" if r["_passed"] else "red"})


def metric_rows(baseline: Optional[dict], candidate: dict) -> list:
    """두 모델의 metrics -> 정상/드리프트 × 4지표 행. 기준(baseline)을 읽지 못했으면 기존·차이는 '없음'."""
    rows = []
    for key, set_name in EVAL_SETS:
        for field, name in METRIC_NAMES:
            new, old = candidate[key][field], None if baseline is None else baseline[key][field]
            rows.append({"평가셋": set_name, "지표": name, "기존": "없음" if old is None else pct(old), "후보": pct(new),
                         "차이(%p)": "없음" if old is None else f"{(new - old) * 100:+.1f}"})
    return rows


def confusion_text(matrix: Optional[list]) -> str:
    """[[TN, FP], [FN, TP]] (양성은 NG) -> 한 줄."""
    if not matrix:
        return "없음"
    (tn, fp), (fn, tp) = matrix
    return f"TN {tn} · FP {fp} · FN {fn} · TP {tp}"


def confusion_rows(baseline: Optional[dict], candidate: dict) -> list:
    return [{"평가셋": set_name, "기존": confusion_text(None if baseline is None else baseline[key]["confusion_matrix"]),
             "후보": confusion_text(candidate[key]["confusion_matrix"])} for key, set_name in EVAL_SETS]


def _duration_text(run: dict) -> str:
    if not run.get("started_at"):
        return "없음"
    if not run.get("finished_at"):
        return "진행 중" if is_run_active(run) else "없음"
    seconds = seconds_between(run["started_at"], parse_utc(run["finished_at"]))
    return "없음" if seconds is None else f"{seconds:.0f}초"


def run_history_rows(items: list) -> list:
    """GET /training/runs 의 items -> 표 행. 소요는 started_at~finished_at 으로 화면에서 계산한 값이다."""
    rows = []
    for run in items:
        gate, snap = _gate(run), run.get("train_data_snapshot")
        if run["status"] != "succeeded" or gate is None:
            gate_text = "없음"
        elif gate.get("baseline_version") is None:
            gate_text = "미적용"
        else:
            gate_text = "통과" if gate.get("passed") else "미통과"
        error = run.get("error_message")
        if error and len(error) > ERROR_PREVIEW_CHARS:
            error = error[:ERROR_PREVIEW_CHARS] + "…"
        rows.append({"실행": run["run_id"], "상태": RUN_STATUS_LABELS.get(run["status"], run["status"]),
                     "시작": local_time_text(run["started_at"]) if run.get("started_at") else "없음", "소요": _duration_text(run),
                     "결과 버전": run.get("result_version") or "없음", "게이트": gate_text,
                     "학습 데이터": f"train {snap.get('train')} · pool {snap.get('pool')}" if snap else "없음", "오류": error or "없음"})
    return rows


def run_history_markdown(rows: list) -> str:
    return md_table(HISTORY_RUN_COLUMNS, rows)


def conflict_message(runs_res: Optional[dict], pool_res: Optional[dict], server_message: str) -> tuple:
    """실행 요청이 409 였을 때. 두 409 는 error 이름이 같아서, 다시 읽은 서버 상태로 구분한다 (메시지 글자 비교는 쓰지 않는다)."""
    if is_run_active(latest_run(runs_res)):
        return "이미 재학습이 실행 중입니다. 현재 진행 상태를 아래에 표시합니다.", "info"
    if pool_res is not None and pool_res["ok"] and pool_res["data"]["total"] == 0:
        return "재학습에 쓸 새 데이터(pool)가 0건입니다. 시뮬레이터로 검사를 쌓은 뒤 다시 눌러 주세요.", "warn"
    return f"재학습 요청이 받아들여지지 않았습니다 (409): {md_cell(server_message)}", "warn"


def start_failure_message(status: int, message: str) -> tuple:
    """실행 요청이 오류 응답(409, 401 제외)으로 끝났을 때. 401 은 화면이 따로 처리한다(로그아웃)."""
    if status == 403:
        return "이 계정에는 재학습 권한이 없습니다.", "error"
    if status == 503:
        return f"고정 데이터셋이 없어 시작하지 못했습니다 ({md_cell(message)})", "error"
    return f"재학습 시작 요청 실패 ({status}): {md_cell(message)}", "error"


# ---------- 승인 화면 (④-4) ----------
# 서버 값은 md_cell 을 거쳐 표시한다. 승인 응답 본문은 쓰지 않고, 처리 뒤에 서버를 다시 읽은 값으로 결과 문장을 만든다.
INFLIGHT_SECONDS = 15  # 보낸 요청 표식이 이보다 오래되면 무시한다 (영구 잠금 방지)
PERMISSION_TEXT = "승인·거절은 관리자만 할 수 있습니다. 목록과 변경 요약은 볼 수 있습니다."
SERVER_DOWN_APPROVAL_TEXT = "요청을 보냈지만 응답을 받지 못했습니다. 처리 여부는 다음 갱신에서 확인됩니다."
BLOCKED_TEXT = "같은 요청을 이미 보내 처리하고 있어 다시 보내지 않았습니다. 처리 여부는 서버를 다시 읽은 아래 값으로 확인하세요."
PENDING_COLUMNS = ("요청", "후보", "요청자", "요청 시각")
APPROVAL_HISTORY_COLUMNS = ("요청", "후보", "상태", "요청자", "요청 시각", "처리자", "처리 시각", "사유")
ACTION_WORDS = {"approve": "승인", "reject": "거절"}


def local_datetime_text(utc_text: Optional[str]) -> str:
    """API 의 UTC 시각 -> 내 시계의 '월-일 시:분:초'. 값이 없으면 '없음'."""
    if not utc_text:
        return "없음"
    try:
        return parse_utc(utc_text).astimezone().strftime("%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return str(utc_text)


def approval_items(approvals_res: Optional[dict]) -> list:
    return [] if approvals_res is None or not approvals_res["ok"] else approvals_res["data"]["items"]


def pending_items(approvals_res: Optional[dict]) -> list:
    """대기(pending) 요청, 서버가 준 순서(최신 먼저) 그대로."""
    return [i for i in approval_items(approvals_res) if i["status"] == "pending"]


def handled_items(approvals_res: Optional[dict]) -> list:
    return [i for i in approval_items(approvals_res) if i["status"] != "pending"]


def find_approval(approvals_res: Optional[dict], approval_id: str) -> Optional[dict]:
    return next((i for i in approval_items(approvals_res) if i["approval_id"] == approval_id), None)


def selected_item(pending: list, chosen_id: Optional[str]) -> Optional[dict]:
    """고른 요청. 목록에 없으면(그 사이 처리됨) 첫 번째, 대기가 없으면 None."""
    return next((i for i in pending if i["approval_id"] == chosen_id), pending[0] if pending else None)


def pending_label(item: dict) -> str:
    return f"{item['approval_id']} · 후보 {item['model_version']} · {local_datetime_text(item['created_at'])} · 요청자 {item['requested_by']}"


def pending_rows(items: list) -> list:
    return [{"요청": i["approval_id"], "후보": i["model_version"], "요청자": i["requested_by"], "요청 시각": local_datetime_text(i["created_at"])} for i in items]


def approval_history_rows(items: list) -> list:
    return [{"요청": i["approval_id"], "후보": i["model_version"], "상태": APPROVAL_LABELS.get(i["status"], i["status"]), "요청자": i["requested_by"],
             "요청 시각": local_datetime_text(i["created_at"]), "처리자": i.get("handled_by") or "없음",
             "처리 시각": local_datetime_text(i.get("handled_at")), "사유": i.get("reason") or "없음"} for i in items]


def candidate_run(runs_res: Optional[dict], version: str) -> Optional[dict]:
    """후보 버전을 만든 학습 기록 (GET /training/runs 에서 result_version 이 같은 것). 게이트 결과는 여기에만 있다."""
    if runs_res is None or not runs_res["ok"]:
        return None
    return next((r for r in runs_res["data"]["items"] if r["status"] == "succeeded" and r.get("result_version") == version), None)


def run_baseline(run: Optional[dict]) -> Optional[str]:
    return (_gate(run) or {}).get("baseline_version")


def production_version(production_res: Optional[dict]) -> Optional[str]:
    """GET /models/production 결과 -> 버전. 운영 모델이 없거나 읽지 못했으면 None."""
    if production_res is None or not production_res["ok"] or production_res["data"] is None:
        return None
    return production_res["data"]["version"]


def baseline_mismatch(run: Optional[dict], production_res: Optional[dict]) -> bool:
    """서버의 기준 모델 검사와 같은 규칙: 게이트가 비교한 기준 != 지금 production 이면 승인은 409. 학습 기록이 없거나 production 을 읽지 못했으면 판단하지 않는다."""
    if run is None or production_res is None or not production_res["ok"]:
        return False
    return run_baseline(run) != production_version(production_res)


def production_line(production_res: Optional[dict]) -> str:
    if production_res is None or not production_res["ok"]:
        return "현재 운영 모델: 읽지 못했습니다"
    version = production_version(production_res)
    return "현재 운영 모델: 없음" if version is None else f"현재 운영 모델: {version}"


def baseline_warning(run: Optional[dict], production_res: Optional[dict]) -> Optional[str]:
    if not baseline_mismatch(run, production_res):
        return None
    return (f"게이트가 비교한 기준 모델({run_baseline(run) or '없음'})과 현재 운영 모델({production_version(production_res) or '없음'})이 달라서 "
            "이 요청을 승인하면 서버가 거절(409)합니다. 거절하거나 재학습으로 새 후보를 만드세요.")


def summary_models(version: str, run: Optional[dict]) -> list:
    """변경 요약에 필요한 모델 상세 [후보, 기준]. 학습 기록을 못 찾으면 후보만."""
    return [v for v in (version, run_baseline(run)) if v]


def can_decide(role: Optional[str]) -> tuple:
    """(버튼 활성 여부, 안내 문구). 비활성은 편의일 뿐이고 서버 403 이 최종이다."""
    return (True, None) if role == "admin" else (False, PERMISSION_TEXT)


def normalize_reason(action: str, text: Optional[str]) -> Optional[str]:
    """입력 사유 -> 보낼 값. 승인은 비어 있으면 보내지 않고(None), 거절은 비어 있어도 그대로 보내 서버 규칙(400)에 맡긴다."""
    cleaned = (text or "").strip()
    if action == "approve":
        return cleaned or None
    return cleaned


def inflight_active(marker: Optional[dict], approval_id: Optional[str], now: float) -> bool:
    """같은 요청을 15초 안에 이미 보냈는가."""
    return marker is not None and marker.get("approval_id") == approval_id and 0 <= now - marker["ts"] < INFLIGHT_SECONDS


def inflight_done(marker: Optional[dict], approvals_res: Optional[dict]) -> bool:
    """표식을 지워도 되는가: 서버를 다시 읽었더니 그 요청이 더 이상 대기가 아니다."""
    if marker is None:
        return False
    item = find_approval(approvals_res, marker["approval_id"])
    return item is not None and item["status"] != "pending"


def failure_message(action: str, status: int, message: str, approvals_res: Optional[dict] = None, approval_id: str = "",
                    production_res: Optional[dict] = None) -> tuple:
    """오류 응답(401, 409 제외). 401 은 화면이 로그아웃으로, 409 는 approval_conflict 로 처리한다. 반환: (문장, 수준)."""
    word = ACTION_WORDS.get(action, action)
    if status == 400 and action == "reject":
        return f"거절 사유를 입력해 주세요 (공백만으로는 안 됩니다). 서버 메시지: {md_cell(message)}", "error"
    if status == 403:
        return "이 계정에는 승인 권한이 없습니다.", "error"
    if status == 404:
        return "승인 요청을 찾을 수 없습니다. 목록을 다시 읽었습니다.", "warn"
    if status == 503:
        item = find_approval(approvals_res, approval_id)
        facts = []
        if item is not None:
            facts.append("요청은 " + APPROVAL_LABELS.get(item["status"], md_cell(item["status"])) + " 상태")
        if production_res is not None and production_res["ok"]:
            facts.append("운영 모델은 " + (production_version(production_res) or "없음"))
        tail = f" (서버를 다시 읽은 값: {', '.join(facts)})" if facts else ""
        return ("새 모델을 불러오지 못해 승인하지 않았습니다. 기존 운영 모델을 계속 사용하고, 요청은 승인 대기로 남아 있어야 합니다" + tail
                + f". 서버 메시지: {md_cell(message)}"), "error"
    return f"{word} 요청 실패 ({status}): {md_cell(message)}", "error"


def approval_conflict(approval_id: str, approvals_res: Optional[dict], runs_res: Optional[dict], production_res: Optional[dict], message: str) -> tuple:
    """409 는 서버를 다시 읽은 상태로 구분한다 (메시지 글자 비교는 쓰지 않는다)."""
    item = find_approval(approvals_res, approval_id)
    if item is not None and item["status"] != "pending":
        who = item.get("handled_by") or "알 수 없음"
        return (f"이미 처리된 요청입니다 ({APPROVAL_LABELS.get(item['status'], md_cell(item['status']))}, 처리자 {md_cell(who)}, "
                f"{local_datetime_text(item.get('handled_at'))}). 현재 상태를 다시 확인했습니다."), "info"
    if item is not None:
        warning = baseline_warning(candidate_run(runs_res, item["model_version"]), production_res)
        if warning:
            return warning, "warn"
    return f"요청이 받아들여지지 않았습니다 (409): {md_cell(message)}", "warn"


def sent_outcome(action: str, approval_id: str, approvals_res: Optional[dict], production_res: Optional[dict]) -> tuple:
    """요청이 성공(2xx)으로 끝난 뒤, 응답 본문이 아니라 다시 읽은 서버 값으로 만든 결과 문장. 반환: (문장, 수준)."""
    item = find_approval(approvals_res, approval_id)
    if item is None:
        return f"{ACTION_WORDS[action]} 요청을 보냈습니다. 서버에서 이 요청을 다시 읽지 못했습니다 (다음 갱신에서 확인됩니다).", "warn"
    version, status = item["model_version"], item["status"]
    if action == "approve" and status == "approved":
        now_prod = production_version(production_res)
        if now_prod == version:
            return f"승인 완료: {version}이 운영 모델입니다 (서버에서 다시 읽은 값).", "ok"
        return f"요청은 승인됨으로 바뀌었지만 지금 운영 모델은 {now_prod or '없음'}입니다 (서버에서 다시 읽은 값).", "warn"
    if action == "reject" and status == "rejected":
        return f"거절 완료: 후보 {version}을 거절했습니다 (서버에서 다시 읽은 값).", "ok"
    return (f"{ACTION_WORDS[action]} 요청을 보냈지만 서버에서는 {APPROVAL_LABELS.get(status, md_cell(status))} 상태입니다. "
            "다음 갱신에서 다시 확인합니다."), "warn"


def action_result(action: str, approval_id: str, result: dict, approvals_res: Optional[dict], runs_res: Optional[dict], production_res: Optional[dict]) -> tuple:
    """버튼 처리 결과 문장. result 는 {"kind": "sent"|"down"|"blocked"|"error", "status", "message"}."""
    kind = result["kind"]
    if kind == "blocked":
        return BLOCKED_TEXT, "info"
    if kind == "down":
        return SERVER_DOWN_APPROVAL_TEXT, "warn"
    if kind == "error":
        if result["status"] == 409:
            return approval_conflict(approval_id, approvals_res, runs_res, production_res, result["message"])
        return failure_message(action, result["status"], result["message"], approvals_res, approval_id, production_res)
    return sent_outcome(action, approval_id, approvals_res, production_res)


# ---------- 모델 관리 화면 (④-5) ----------
# 모델 목록(GET /models)에는 지표가 들어 있어 stage 와 지표가 같은 응답에서 온다. 롤백 응답 본문은 쓰지 않고, 처리 뒤에 서버를 다시 읽은 값으로 결과 문장을 만든다.
DEPLOYMENT_LIMIT = 20  # 배포 이력을 읽는 최근 건수 (GET /deployments?limit=20, 허용 범위 1~1000)
STAGE_LABELS = {"candidate": "후보", "staging": "승인 대기(staging)", "production": "운영 중", "archived": "보관(archived)", "rejected": "거절됨"}
DEPLOYMENT_ACTION_LABELS = {"deploy": "배포", "rollback": "롤백"}
MODEL_COLUMNS = ("버전", "단계", "생성 시각", "마지막으로 운영에 올린 사람", "올린 시각")
DEPLOYMENT_COLUMNS = ("번호", "시각", "버전", "종류", "처리자", "사유")
ROLLBACK_PERMISSION_TEXT = "롤백은 관리자만 할 수 있습니다. 버전 목록, 지표 비교, 배포 이력은 볼 수 있습니다."
NO_ARCHIVED_TEXT = "롤백할 수 있는(archived) 버전이 없습니다."
SERVER_DOWN_ROLLBACK_TEXT = "요청을 보냈지만 응답을 받지 못했습니다. 처리 여부는 다음 갱신에서 확인됩니다."
ROLLBACK_BLOCKED_TEXT = "같은 롤백 요청을 이미 보내 처리하고 있어 다시 보내지 않았습니다. 처리 여부는 서버를 다시 읽은 아래 값으로 확인하세요."


def model_items(models_res: Optional[dict]) -> list:
    """GET /models 결과의 항목, 서버가 준 순서(최신 먼저) 그대로. 조회 실패면 빈 목록."""
    return [] if models_res is None or not models_res["ok"] else models_res["data"]["items"]


def find_model(models_res: Optional[dict], version: Optional[str]) -> Optional[dict]:
    return next((m for m in model_items(models_res) if m["version"] == version), None)


def model_versions(models_res: Optional[dict]) -> list:
    return [m["version"] for m in model_items(models_res)]


def production_of(models_res: Optional[dict]) -> Optional[str]:
    """목록에서 stage 가 production 인 버전. 없거나 읽지 못했으면 None."""
    return next((m["version"] for m in model_items(models_res) if m["stage"] == "production"), None)


def stage_label(stage: str) -> str:
    return STAGE_LABELS.get(stage, str(stage))


def model_rows(items: list) -> list:
    return [{"버전": m["version"], "단계": stage_label(m["stage"]), "생성 시각": local_datetime_text(m["created_at"]),
             "마지막으로 운영에 올린 사람": m.get("approved_by") or "없음", "올린 시각": local_datetime_text(m.get("approved_at")),
             "_production": m["stage"] == "production"} for m in items]


def model_table_markdown(rows: list) -> str:
    """운영 중인 행은 단계 칸을 초록으로 강조한다."""
    return md_table(MODEL_COLUMNS, rows, tint={"단계": lambda r: "green" if r["_production"] else None})


def rollback_targets(models_res: Optional[dict]) -> list:
    """롤백 대상으로 고를 수 있는 버전: archived 만 (서버도 그 밖은 409). 서버가 준 순서 그대로."""
    return [m["version"] for m in model_items(models_res) if m["stage"] == "archived"]


def default_compare(models_res: Optional[dict]) -> tuple:
    """(A, B) 기본값. A = 운영 모델(없으면 가장 최신), B = A 를 뺀 가장 최신 버전. 없으면 None."""
    versions = model_versions(models_res)
    a = production_of(models_res) or (versions[0] if versions else None)
    b = next((v for v in versions if v != a), None)
    return a, b


def compare_labels(a: str, b: str) -> tuple:
    return f"{a} (A)", f"{b} (B)"


def compare_columns(a: str, b: str) -> tuple:
    return ("평가셋", "지표") + compare_labels(a, b) + ("차이(B−A, %p)",)


def _rekey(rows: list, names: dict) -> list:
    return [{names.get(k, k): v for k, v in row.items()} for row in rows]


def compare_rows(a: str, a_metrics: dict, b: str, b_metrics: dict) -> list:
    """두 버전의 지표 비교 행. 계산은 metric_rows 를 그대로 쓰고 머리글만 버전 이름으로 바꾼다 (차이 = B − A)."""
    label_a, label_b = compare_labels(a, b)
    names = {"기존": label_a, "후보": label_b, "차이(%p)": "차이(B−A, %p)"}
    return _rekey(metric_rows(a_metrics, b_metrics), names)


def compare_confusion_columns(a: str, b: str) -> tuple:
    return ("평가셋",) + compare_labels(a, b)


def compare_confusion_rows(a: str, a_metrics: dict, b: str, b_metrics: dict) -> list:
    label_a, label_b = compare_labels(a, b)
    return _rekey(confusion_rows(a_metrics, b_metrics), {"기존": label_a, "후보": label_b})


SINGLE_METRIC_COLUMNS = ("평가셋", "지표", "값")


def single_metric_rows(metrics: dict) -> list:
    """버전이 하나뿐일 때: 비교 없이 그 버전의 정상/드리프트 × 4지표."""
    return [{"평가셋": r["평가셋"], "지표": r["지표"], "값": r["후보"]} for r in metric_rows(None, metrics)]


def deployment_items(deployments_res: Optional[dict]) -> list:
    return [] if deployments_res is None or not deployments_res["ok"] else deployments_res["data"]["items"]


def deployment_rows(items: list) -> list:
    return [{"번호": d["deployment_id"], "시각": local_datetime_text(d["at"]), "버전": d["version"],
             "종류": DEPLOYMENT_ACTION_LABELS.get(d["action"], d["action"]), "처리자": d["actor"], "사유": d.get("reason") or "없음"} for d in items]


def can_rollback(role: Optional[str]) -> tuple:
    """(버튼 활성 여부, 안내 문구). 비활성은 편의일 뿐이고 서버 403 이 최종이다."""
    return (True, None) if role == "admin" else (False, ROLLBACK_PERMISSION_TEXT)


def normalize_rollback_reason(text: Optional[str]) -> str:
    """입력 사유 -> 보낼 값. 비어 있어도 그대로 보내 서버 규칙(400)에 맡긴다."""
    return (text or "").strip()


def rollback_inflight_active(marker: Optional[dict], version: Optional[str], now: float) -> bool:
    """같은 버전의 롤백을 15초 안에 이미 보냈는가."""
    return marker is not None and marker.get("version") == version and 0 <= now - marker["ts"] < INFLIGHT_SECONDS


def rollback_inflight_done(marker: Optional[dict], models_res: Optional[dict]) -> bool:
    """표식을 지워도 되는가: 서버를 다시 읽었더니 그 버전이 더 이상 archived 가 아니다."""
    if marker is None:
        return False
    item = find_model(models_res, marker["version"])
    return item is not None and item["stage"] != "archived"


def rollback_failure_message(status: int, message: str, models_res: Optional[dict] = None, version: str = "") -> tuple:
    """오류 응답(401, 409 제외). 401 은 화면이 로그아웃으로, 409 는 rollback_conflict 로 처리한다. 반환: (문장, 수준)."""
    if status == 400:
        return f"롤백 사유를 입력해 주세요 (공백만으로는 안 되고 500자 이내). 서버 메시지: {md_cell(message)}", "error"
    if status == 403:
        return "이 계정에는 롤백 권한이 없습니다.", "error"
    if status == 404:
        return "롤백할 버전을 찾을 수 없습니다. 목록을 다시 읽었습니다.", "warn"
    if status == 503:
        now_prod = production_of(models_res)
        tail = f" (서버를 다시 읽은 값: 운영 모델은 {now_prod or '없음'})" if models_res is not None and models_res["ok"] else ""
        return (f"모델 {md_cell(version)}을 불러오지 못해 롤백하지 않았습니다. 기존 운영 모델을 계속 사용합니다{tail}. "
                f"서버 메시지: {md_cell(message)}"), "error"
    return f"롤백 요청 실패 ({status}): {md_cell(message)}", "error"


def rollback_conflict(version: str, models_res: Optional[dict], message: str) -> tuple:
    """409 는 서버를 다시 읽은 stage 로 구분한다 (메시지 글자 비교는 쓰지 않는다)."""
    item = find_model(models_res, version)
    if item is not None and item["stage"] == "production":
        return f"{md_cell(version)}은 이미 운영 중인 버전입니다. 현재 상태를 다시 확인했습니다.", "info"
    if item is not None:
        return f"{md_cell(version)}은 지금 {stage_label(item['stage'])} 상태라서 롤백할 수 없습니다. 현재 상태를 다시 확인했습니다.", "warn"
    return f"요청이 받아들여지지 않았습니다 (409): {md_cell(message)}", "warn"


def rollback_outcome(version: str, models_res: Optional[dict]) -> tuple:
    """요청이 성공(2xx)으로 끝난 뒤, 응답 본문이 아니라 다시 읽은 서버 값으로 만든 결과 문장. 반환: (문장, 수준)."""
    item = find_model(models_res, version)
    if item is None:
        return "롤백 요청을 보냈습니다. 서버에서 이 버전을 다시 읽지 못했습니다 (다음 갱신에서 확인됩니다).", "warn"
    if item["stage"] == "production":
        return f"롤백 완료: {md_cell(version)}이 운영 모델입니다 (서버에서 다시 읽은 값).", "ok"
    return (f"롤백 요청을 보냈지만 서버에서는 {md_cell(version)}이 {stage_label(item['stage'])} 상태입니다 "
            f"(운영 모델: {production_of(models_res) or '없음'}). 다음 갱신에서 다시 확인합니다."), "warn"


def rollback_result(version: str, result: dict, models_res: Optional[dict]) -> tuple:
    """버튼 처리 결과 문장. result 는 {"kind": "sent"|"down"|"blocked"|"error", "status", "message"}."""
    kind = result["kind"]
    if kind == "blocked":
        return ROLLBACK_BLOCKED_TEXT, "info"
    if kind == "down":
        return SERVER_DOWN_ROLLBACK_TEXT, "warn"
    if kind == "error":
        if result["status"] == 409:
            return rollback_conflict(version, models_res, result["message"])
        return rollback_failure_message(result["status"], result["message"], models_res, version)
    return rollback_outcome(version, models_res)
