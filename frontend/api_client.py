"""백엔드 API 호출 (HTTP 를 부르는 유일한 곳).

- 응답은 JSON 을 그대로 dict 로 돌려준다 (backend 코드를 가져다 쓰지 않는다).
- 오류는 두 가지로 바꿔 던진다: 서버가 꺼졌거나 응답이 없으면 ServerDown, 서버가 오류로 답하면 ApiError(상태 코드, error 이름, 설명).
- client 를 밖에서 넣을 수 있다 (시험에서 가짜 전송이나 Starlette TestClient 를 넣는다. TestClient 는 httpx2.Client 의 하위 클래스다).
"""
from typing import Optional

import httpx2

from frontend import config


SLOW_READ_TIMEOUT_SECONDS = 8.0  # stop(서버가 최대 5초 기다림)·generate·predict(5초 요건)용 읽기 제한 시간


class ApiError(Exception):
    """서버가 오류 응답을 줬다 (docs/api-v1.md 1장 오류 형식 {"error", "message"})."""

    def __init__(self, status: int, error: str, message: str):
        super().__init__(f"{status} {error}: {message}")
        self.status = status
        self.error = error
        self.message = message


class ServerDown(Exception):
    """서버에 연결할 수 없거나 응답이 없다."""


class ApiClient:
    def __init__(self, base_url: Optional[str] = None, token: Optional[str] = None, client: Optional[httpx2.Client] = None):
        self.base_url = base_url or config.api_url()
        self.token = token
        self._owns_client = client is None  # 우리가 만든 연결만 우리가 닫는다
        self._client = client if client is not None else httpx2.Client(
            base_url=self.base_url,
            timeout=httpx2.Timeout(config.READ_TIMEOUT_SECONDS, connect=config.CONNECT_TIMEOUT_SECONDS))

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "ApiClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _send(self, method: str, path: str, params: Optional[dict], json: Optional[dict], timeout: Optional[float]):
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        extra = {} if timeout is None else {"timeout": httpx2.Timeout(timeout, connect=config.CONNECT_TIMEOUT_SECONDS)}
        try:
            response = self._client.request(method, path, params=params, json=json, headers=headers, **extra)
        except httpx2.TransportError as e:  # 연결 실패, 제한 시간 초과 등
            raise ServerDown(f"{type(e).__name__}: {e}") from e
        if response.status_code >= 400:
            raise self._error(response)
        return response

    def _request(self, method: str, path: str, *, params: Optional[dict] = None, json: Optional[dict] = None,
                 timeout: Optional[float] = None):
        """timeout 은 읽기 제한 시간(초). 생략하면 기본값(config.READ_TIMEOUT_SECONDS)."""
        return self._send(method, path, params, json, timeout).json()

    def _request_bytes(self, method: str, path: str, *, timeout: Optional[float] = None) -> bytes:
        return self._send(method, path, None, None, timeout).content

    @staticmethod
    def _error(response) -> ApiError:
        try:
            body = response.json()
            return ApiError(response.status_code, str(body["error"]), str(body["message"]))
        except (ValueError, KeyError, TypeError):
            return ApiError(response.status_code, "unknown", response.text[:200] or "설명 없음")

    def login(self, username: str, password: str) -> dict:
        """POST /auth/login -> {"access_token", "token_type", "expires_in", "role", "username"}."""
        return self._request("POST", "/auth/login", json={"username": username, "password": password})

    def health(self) -> dict:
        """GET /health (공개) -> {"status", "database", "model": {"production_version", "loaded"}}."""
        return self._request("GET", "/health")

    # ---------- 대시보드 조회 (모두 GET, 로그인 필요. /health 만 공개) ----------

    @staticmethod
    def _params(**values) -> dict:
        return {k: v for k, v in values.items() if v is not None}

    def simulator_status(self) -> dict:
        """GET /simulator/status -> 9개 필드 (api-v1.md 3-2 상태 형식)."""
        return self._request("GET", "/simulator/status")

    def training_runs(self, limit: int = 1) -> dict:
        """GET /training/runs -> {"items", "total"} (최신 먼저)."""
        return self._request("GET", "/training/runs", params={"limit": limit})

    def production_model(self) -> dict:
        """GET /models/production. 운영 모델이 없으면 ApiError(404)."""
        return self._request("GET", "/models/production")

    def inspections(self, since: Optional[str] = None, data_split: Optional[str] = None, limit: int = 1) -> dict:
        """GET /inspections -> {"items", "total"}. since 는 시간대가 있는 ISO 문자열 (`+` 는 자동으로 %2B 로 보낸다)."""
        return self._request("GET", "/inspections", params=self._params(since=since, data_split=data_split, limit=limit))

    def prediction_stats(self) -> dict:
        """GET /predictions/stats -> total, ng_count, ng_rate, recent."""
        return self._request("GET", "/predictions/stats")

    def performance(self, bucket_seconds: int = 3600) -> dict:
        """GET /monitoring/performance. 카드는 recent 와 criteria 만 쓰므로 구간을 크게 잡아 points 를 작게 받는다."""
        return self._request("GET", "/monitoring/performance", params={"bucket_seconds": bucket_seconds})

    def approvals(self, status: Optional[str] = None, limit: int = 1) -> dict:
        """GET /approvals -> {"items", "total"}."""
        return self._request("GET", "/approvals", params=self._params(status=status, limit=limit))

    def drift(self) -> dict:
        """GET /monitoring/drift -> {"performance_warning", "input_distribution"}."""
        return self._request("GET", "/monitoring/drift")

    def deployments(self, limit: int = 1) -> dict:
        """GET /deployments -> {"items", "total"} (최신 먼저)."""
        return self._request("GET", "/deployments", params={"limit": limit})

    # ---------- 재학습 화면 (④-3) ----------
    # POST /training/runs 는 queued 한 건을 바로 돌려주고 학습은 서버가 뒤에서 한다 (api-v1.md A16). 읽기 제한 시간은 기본값을 쓴다.

    def start_training_run(self) -> dict:
        """POST /training/runs (201, queued 한 건). 이미 실행 중이거나 pool 이 0건이면 ApiError(409), 고정 데이터셋이 없으면 503."""
        return self._request("POST", "/training/runs", json={"trigger": "manual"})

    def model(self, version: str) -> dict:
        """GET /models/{version} -> 모델 상세 7개 필드 (metrics 포함). 없는 버전은 ApiError(404)."""
        return self._request("GET", f"/models/{version}")

    # ---------- 승인 화면 (④-4) ----------
    # 승인은 새 모델을 불러온 뒤에 응답한다 (모델 없이 켠 서버의 첫 승인은 약 5.7초). 기본 3초로는 정상 동작이 ServerDown 으로 보이므로 8초로 늘린다.
    # 거절은 모델을 읽지 않지만 승인과 한 번에 하나씩 처리되어 기다릴 수 있으므로 같은 8초를 쓴다.

    def approve_approval(self, approval_id: str, reason: Optional[str] = None) -> dict:
        """POST /approvals/{id}/approve (200). 사유는 선택이라 없으면 빈 본문 {} 을 보낸다. 관리자만 (운영자는 403)."""
        body = {} if reason is None else {"reason": reason}
        return self._request("POST", f"/approvals/{approval_id}/approve", json=body, timeout=SLOW_READ_TIMEOUT_SECONDS)

    def reject_approval(self, approval_id: str, reason: str) -> dict:
        """POST /approvals/{id}/reject (200). 사유는 필수 (빠지거나 공백이면 400). 관리자만 (운영자는 403)."""
        return self._request("POST", f"/approvals/{approval_id}/reject", json={"reason": reason}, timeout=SLOW_READ_TIMEOUT_SECONDS)

    # ---------- 모델 관리 화면 (④-5) ----------
    # 롤백도 대상 모델을 새로 읽은 뒤에 응답한다 (모델 없이 켠 서버의 첫 롤백 시간은 미측정). 승인과 같은 이유로 8초를 쓴다.

    def models(self, stage: Optional[str] = None) -> dict:
        """GET /models -> {"items", "total"} (최신 먼저). 항목은 모델 상세 7개 필드(metrics 포함)와 같다."""
        return self._request("GET", "/models", params=self._params(stage=stage))

    def rollback_model(self, version: str, reason: str) -> dict:
        """POST /models/{version}/rollback (200). 사유는 필수 (빠지거나 공백이면 400). archived 가 아니면 409, 모델을 못 불러오면 503. 관리자만 (운영자는 403)."""
        return self._request("POST", f"/models/{version}/rollback", json={"reason": reason}, timeout=SLOW_READ_TIMEOUT_SECONDS)

    # ---------- 검사/추론 화면 (④-2) ----------
    # stop 은 서버가 최대 5초 기다리고 predict 는 5초 요건이라, 기본 3초 제한으로는 정상 동작이 ServerDown 으로 보인다.
    # 그래서 stop·generate·predict 만 8초로 늘린다.

    def simulator_start(self, defect_rate: float, interval_seconds: float) -> dict:
        """POST /simulator/start (200, 상태 형식). 이미 실행 중이면 ApiError(409)."""
        return self._request("POST", "/simulator/start", json={"defect_rate": defect_rate, "interval_seconds": interval_seconds})

    def simulator_stop(self) -> dict:
        """POST /simulator/stop (200, 멈춰 있어도 200)."""
        return self._request("POST", "/simulator/stop", timeout=SLOW_READ_TIMEOUT_SECONDS)

    def simulator_generate(self, scenario: str = "normal") -> dict:
        """POST /simulator/generate (201, 검사 8개 필드)."""
        return self._request("POST", "/simulator/generate", json={"scenario": scenario}, timeout=SLOW_READ_TIMEOUT_SECONDS)

    def simulator_drift(self, enabled: bool, drift_type: Optional[str] = None) -> dict:
        """POST /simulator/drift. 끌 때는 drift_type 을 보내지 않는다 (보내면 400)."""
        body = {"enabled": enabled} if drift_type is None else {"enabled": enabled, "drift_type": drift_type}
        return self._request("POST", "/simulator/drift", json=body)

    def predict(self, inspection_id: str) -> dict:
        """POST /predict (201, prediction 8개 필드)."""
        return self._request("POST", "/predict", json={"inspection_id": inspection_id}, timeout=SLOW_READ_TIMEOUT_SECONDS)

    def predictions(self, limit: int = 20) -> dict:
        """GET /predictions -> {"items", "total"} (최신 먼저, 항목에 true_label·correct 포함)."""
        return self._request("GET", "/predictions", params={"limit": limit})

    def inspection(self, inspection_id: str) -> dict:
        """GET /inspections/{id} -> 검사 8개 필드 + predictions."""
        return self._request("GET", f"/inspections/{inspection_id}")

    def inspection_image(self, inspection_id: str) -> bytes:
        """GET /inspections/{id}/image -> PNG 바이트. 없으면 ApiError(404)."""
        return self._request_bytes("GET", f"/inspections/{inspection_id}/image")
