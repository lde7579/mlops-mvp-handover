"""자동 생성: POST /simulator/start·stop·drift 의 본체 (api-v1.md A23, 4장 A26).

켜면 스레드 하나가 "검사 1건 생성 -> 바로 판정 -> 간격만큼 대기"를 반복한다.
- 생성과 판정은 밖에서 받은 함수다 (generate, judge). 시험은 가짜를, 서버는 build_runner 가 만든 진짜를 넣는다.
- 판정이 실패해도(운영 모델 없음 등) 검사는 남기고 판정만 건너뛴다 (judge_failures, last_error). 생성은 계속한다.
- 생성이 연속 MAX_GENERATE_FAILURES 번 실패하면 스스로 멈춘다 (running=False, last_error 유지).
- 드리프트 주입 상태(켬/끔, 종류)는 서버 메모리에만 있다. 서버를 다시 켜면 꺼진 상태다. 강도는 수집기 설정(collector.toml)이 정한다.
- 불량률은 시드를 고정한 난수(LABEL_SEED)로 OK/NG 를 정해 수집기의 label 옵션으로 넘긴다 (같은 설정이면 같은 순서).
- 서버가 꺼질 때 shutdown() 으로 스레드를 멈춘다 (대기는 Event.wait 라서 멈춤 신호에 바로 깬다).
- 프로세스 하나(워커 1개)만 지원한다. 상태 확인과 변경은 락 하나로 한다.
"""
import logging
import random
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from backend.database import session_scope
from backend.generation import GenerationService
from backend.inference import InferenceService
from backend.judging import judge_inspection
from backend.repository import ConflictError
from common.schemas import Inspection

log = logging.getLogger("backend")

DEFAULT_DEFECT_RATE = 0.2
DEFAULT_INTERVAL_SECONDS = 2.0
LABEL_SEED = 42
MAX_GENERATE_FAILURES = 5
JOIN_TIMEOUT_SECONDS = 5.0
MAX_MESSAGE = 300
ALREADY_RUNNING = "이미 실행 중입니다"


class SimulatorBusyError(ConflictError):
    """자동 생성이 이미 실행 중이다 (409)."""


@dataclass(frozen=True)
class SimulatorStatus:
    running: bool
    interval_seconds: Optional[float]
    defect_rate: Optional[float]
    drift_enabled: bool
    drift_type: Optional[str]
    generated: int
    judged: int
    judge_failures: int
    last_error: Optional[str]
    started_at: Optional[datetime]


def _default_wait(stop: threading.Event, seconds: float) -> bool:
    """seconds 만큼 기다리다 멈춤 신호가 오면 바로 True (멈춰야 함)."""
    return stop.wait(seconds)


def _message(prefix: str, error: Exception) -> str:
    return f"{prefix}: {(str(error).strip() or type(error).__name__)}"[:MAX_MESSAGE]


class SimulatorRunner:
    def __init__(self, generate: Callable[[str, str, Optional[str]], Inspection], judge: Callable[[Inspection], None], *,
                 clock: Optional[Callable[[], datetime]] = None,
                 wait: Callable[[threading.Event, float], bool] = _default_wait,
                 join_timeout: float = JOIN_TIMEOUT_SECONDS):
        """generate(scenario, label, drift_type) -> Inspection, judge(inspection) -> None.
        wait(멈춤 신호, 초) -> 멈춰야 하면 True. 시험에서 가짜로 바꿔 끼운다."""
        self._generate = generate
        self._judge = judge
        self._clock = clock if clock is not None else (lambda: datetime.now(timezone.utc))
        self._wait = wait
        self._join_timeout = join_timeout
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._running = False
        self._interval: Optional[float] = None
        self._defect_rate: Optional[float] = None
        self._started_at: Optional[datetime] = None
        self._drift_enabled = False
        self._drift_type: Optional[str] = None
        self._generated = 0
        self._judged = 0
        self._judge_failures = 0
        self._last_error: Optional[str] = None

    # ---------- 요청 안 ----------

    @property
    def thread(self) -> Optional[threading.Thread]:
        """마지막으로 시작한 스레드 (시험이 끝났는지 확인하는 용도)."""
        return self._thread

    def status(self) -> SimulatorStatus:
        with self._lock:
            return self._status()

    def _status(self) -> SimulatorStatus:
        return SimulatorStatus(
            running=self._running, interval_seconds=self._interval if self._running else None,
            defect_rate=self._defect_rate if self._running else None, drift_enabled=self._drift_enabled,
            drift_type=self._drift_type, generated=self._generated, judged=self._judged,
            judge_failures=self._judge_failures, last_error=self._last_error,
            started_at=self._started_at if self._running else None)

    def start(self, defect_rate: float = DEFAULT_DEFECT_RATE, interval_seconds: float = DEFAULT_INTERVAL_SECONDS) -> SimulatorStatus:
        """자동 생성을 시작한다. 이미 실행 중이면 SimulatorBusyError. 건수와 마지막 오류는 새로 센다."""
        with self._lock:
            if self._running:
                raise SimulatorBusyError(ALREADY_RUNNING)
            self._stop = threading.Event()
            self._running = True
            self._interval, self._defect_rate = interval_seconds, defect_rate
            self._started_at = self._clock()
            self._generated = self._judged = self._judge_failures = 0
            self._last_error = None
            self._thread = threading.Thread(
                target=self._loop, args=(self._stop, interval_seconds, defect_rate, random.Random(LABEL_SEED)),
                name="simulator-runner", daemon=True)
            self._thread.start()
            return self._status()

    def stop(self) -> SimulatorStatus:
        """자동 생성을 멈추고 스레드가 끝나길 기다린다 (최대 join_timeout 초). 실행 중이 아니어도 된다."""
        with self._lock:
            thread, stop = self._thread, self._stop
        stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(self._join_timeout)
            if thread.is_alive():
                log.warning("자동 생성 스레드가 %.1f초 안에 끝나지 않았습니다 (판정 중일 수 있습니다)", self._join_timeout)
        return self.status()

    def shutdown(self) -> None:
        """서버가 꺼질 때: 스레드를 멈춘다. 어떤 일이 있어도 서버 종료를 막지 않는다."""
        try:
            self.stop()
        except Exception:
            log.exception("자동 생성 스레드를 멈추는 중 오류가 났습니다")

    def set_drift(self, enabled: bool, drift_type: Optional[str] = None) -> SimulatorStatus:
        """드리프트 주입을 켜거나 끈다. 자동 생성이 꺼져 있어도 되고, 다음 start 부터 적용된다."""
        with self._lock:
            self._drift_enabled = enabled
            self._drift_type = drift_type if enabled else None
            return self._status()

    # ---------- 스레드 ----------

    def _loop(self, stop: threading.Event, interval: float, defect_rate: float, rng: random.Random) -> None:
        streak = 0  # 연속 생성 실패 횟수
        try:
            while not stop.is_set():
                streak = streak + 1 if not self._cycle(defect_rate, rng) else 0
                if streak >= MAX_GENERATE_FAILURES:
                    log.error("검사 생성이 연속 %d번 실패해 자동 생성을 멈춥니다", streak)
                    break
                if self._wait(stop, interval):
                    break
        finally:
            with self._lock:
                self._running = False

    def _cycle(self, defect_rate: float, rng: random.Random) -> bool:
        """한 바퀴: 생성 -> 판정. 생성에 성공했으면 True (판정 실패는 성공으로 친다)."""
        with self._lock:
            drift_type = self._drift_type if self._drift_enabled else None
        label = "NG" if rng.random() < defect_rate else "OK"
        try:
            inspection = self._generate("drift" if drift_type else "normal", label, drift_type)
        except Exception as e:
            log.exception("자동 생성: 검사를 만들지 못했습니다")
            with self._lock:
                self._last_error = _message("생성 실패", e)
            return False
        with self._lock:
            self._generated += 1
        try:
            self._judge(inspection)
        except Exception as e:  # 운영 모델 없음(ModelUnavailableError), 이미지 문제(JudgeError) 등: 판정만 건너뛴다
            log.warning("자동 판정을 건너뜁니다 (%s): %s", inspection.inspection_id, e)
            with self._lock:
                self._judge_failures += 1
                self._last_error = _message("판정 건너뜀", e)
        else:
            with self._lock:
                self._judged += 1
        return True


def build_runner(settings, session_factory, generation: GenerationService, inference: InferenceService,
                 **runner_args) -> SimulatorRunner:
    """서버용 러너: 생성과 판정을 각각 짧은 별도 트랜잭션으로 한다 (판정하는 동안 쓰기 잠금을 쥐지 않는다).
    runner_args 는 SimulatorRunner 에 그대로 넘긴다 (시험이 가짜 wait 를 넣을 때 쓴다)."""

    def generate(scenario: str, label: str, drift_type: Optional[str]) -> Inspection:
        with session_scope(session_factory) as session:
            return generation.generate(session, scenario, label=label, drift_type=drift_type)

    def judge(inspection: Inspection) -> None:
        with session_scope(session_factory) as session:
            judge_inspection(session, inference, settings.root, inspection)

    return SimulatorRunner(generate, judge, **runner_args)
