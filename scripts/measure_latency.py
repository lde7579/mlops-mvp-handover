"""API 포함 판정 시간(5초 요건) 측정: POST /predict 요청 전체 시간(B)과 모델 계산 시간 latency_ms(A)를 잰다.

실행:  python -m scripts.measure_latency [--quick] [--report data/latency_report.txt | -]

- 판정 기준: 모든 /predict 요청의 (B) 가 5000ms 이하 (docs/requirements.md "API 포함 5초 측정", 계약 3장 latency_ms).
  (C) 화면에 표시되기까지의 시간은 화면이 있어야 하므로 이 스크립트의 범위 밖이다.
- 실제 data/mlops.db, models/v1, data/pool 은 건드리지 않는다. 임시 폴더에 실제 models/v1 의 **복사본**, 임시 DB, 수집기로 만든 검사 20건을
  만들고, 임시 포트로 실제 uvicorn 서버를 띄워 http.client 로 잰다. 끝나면 서버를 PID 로 종료하고 포트가 닫혔는지 확인한다.
- 항목: 1 서버 시작 -> loaded=true, 2 콜드/웜 /predict, 3 generate -> predict 연속, 4 동시 요청(2·4·8개),
  5 요청 전체 시간 분해, 추가 (유휴 후 첫 요청, 지연 로딩 첫 /predict = 판정 대상 아님).
- 항목 5 의 의존: 제품 함수(InspectionRepository.get, PredictionRepository.add, Session.commit 등)를 **메모리에서만** 감싸 시간을 잰다.
  제품 내부 이름이 바뀌면 항목 5 만 경고하고 건너뛴다 (1~4 판정은 계속된다). 이 의존은 TIMER_TARGETS 한 곳에 있다.
- 요구: models/v1 (python -m scripts.train_v1), torch, backend/ (조원 전달 압축에 backend/ 가 들어간 뒤에 실행 가능).
- 모델이나 전처리를 바꾼 뒤에는 다시 실행해서 5초 요건을 확인한다. 보고서가 이미 있으면 덮어쓰지 않는다 (--overwrite).
- 무거운 모듈(torch, backend, uvicorn)은 측정 함수 안에서만 가져온다: 이 파일을 가져오는 것만으로는 아무것도 올라오지 않는다.
"""
import argparse
import contextvars
import hashlib
import http.client
import json
import os
import secrets
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIMIT_MS = 5000.0
N_INSPECTIONS = 20
ENV_ROOT, ENV_SECRET = "MLOPS_LATENCY_ROOT", "MLOPS_LATENCY_SECRET"
DEFAULT_REPORT = "data/latency_report.txt"
CONCURRENCY_FULL = ((4, 75), (2, 100), (8, 25))
CONCURRENCY_QUICK = ((2, 5), (4, 3), (8, 2))

# 항목 5 가 의존하는 제품 내부 이름: (모듈 경로, 소유자 이름, 속성 이름, 구간 이름)
TIMER_TARGETS = (
    ("backend.deps", None, "decode_token", "auth"),
    ("backend.repository", "InspectionRepository", "get", "get_inspection"),
    ("backend.routers.predict", None, "resolve_image", "resolve_image"),
    ("backend.routers.predict", None, "read_image", "read_image"),
    ("backend.inference", "InferenceService", "ensure_loaded", "ensure_loaded"),
    ("backend.inference", "InferenceService", "predict", "inference_predict"),
    ("backend.repository", "PredictionRepository", "add", "prediction_add"),
    ("sqlalchemy.orm", "Session", "commit", "commit"),
)


# ------------------------------------------------------------------ 통계·판정 (순수 함수)
def percentile(values, p):
    """가까운 순위(nearest-rank) 백분위. values 가 비어 있으면 ValueError."""
    if not values:
        raise ValueError("percentile: 값이 하나도 없습니다")
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * p // 100))  # 올림
    return ordered[int(rank) - 1]


def margin_pct(max_ms, limit=LIMIT_MS):
    """한도 대비 여유율(%). 한도를 넘으면 음수."""
    return (limit - max_ms) / limit * 100


def summarize(values):
    if not values:
        return None
    return {"n": len(values), "min": min(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "p95": percentile(values, 95), "p99": percentile(values, 99), "max": max(values), "margin_pct": margin_pct(max(values))}


def verdict(max_ms, limit=LIMIT_MS):
    """정확히 한도(5000ms)면 통과, 넘으면 초과."""
    return max_ms <= limit


def fmt(s):
    if not s:
        return "(자료 없음)"
    return (f"n={s['n']} 최소 {s['min']:.1f} 평균 {s['mean']:.1f} 중앙값 {s['median']:.1f} p95 {s['p95']:.1f} p99 {s['p99']:.1f} "
            f"최댓값 {s['max']:.1f}ms (5000ms 대비 여유 {s['margin_pct']:.1f}%)")


# ------------------------------------------------------------------ 인자·경로 검사 (순수 함수)
def _positive(text):
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"1 이상의 정수여야 합니다 (받은 값: {text!r})") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"1 이상의 정수여야 합니다 (받은 값: {text!r})")
    return value


def _non_negative(text):
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"0 이상의 정수여야 합니다 (받은 값: {text!r})") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"0 이상의 정수여야 합니다 (받은 값: {text!r})")
    return value


def build_parser():
    p = argparse.ArgumentParser(prog="python -m scripts.measure_latency",
                                description="API 포함 판정 시간(5초 요건)을 임시 루트·임시 포트의 실제 서버로 잰다")
    p.add_argument("--restarts", type=_positive, default=None, help="서버를 새로 띄우는 횟수 (기본 5, --quick 이면 1)")
    p.add_argument("--warm", type=_positive, default=None, help="웜 연속 /predict 횟수 (기본 300, --quick 이면 20)")
    p.add_argument("--pairs", type=_positive, default=None, help="generate -> predict 쌍 수 (기본 200, --quick 이면 10)")
    p.add_argument("--lazy", type=_positive, default=None, help="지연 로딩 첫 판정 반복 횟수 (기본 3, --quick 이면 1)")
    p.add_argument("--idle-seconds", type=_non_negative, default=None, help="유휴 후 첫 요청 전 쉬는 초 (기본 30, 0 이면 생략, --quick 이면 생략)")
    p.add_argument("--quick", action="store_true", help="작은 규모로 끝까지 한 번만 확인한다 (판정 통계로는 쓰지 않는다)")
    p.add_argument("--report", default=DEFAULT_REPORT, help=f"보고서 경로(프로젝트 루트 안). '-' 이면 화면에만 출력 (기본 {DEFAULT_REPORT})")
    p.add_argument("--overwrite", action="store_true", help="보고서가 이미 있어도 덮어쓴다")
    return p


def settings_from_args(args):
    """옵션 값을 채운다 (--quick 이면 작은 값이 기본)."""
    quick = args.quick
    pick = lambda value, full, small: value if value is not None else (small if quick else full)
    return {
        "restarts": pick(args.restarts, 5, 1), "warm": pick(args.warm, 300, 20), "pairs": pick(args.pairs, 200, 10),
        "lazy": pick(args.lazy, 3, 1), "idle_seconds": pick(args.idle_seconds, 30, 0),
        "levels": CONCURRENCY_QUICK if quick else CONCURRENCY_FULL, "quick": quick,
    }


def assert_safe_root(path, project_root=PROJECT_ROOT):
    """측정용 임시 루트가 프로젝트 안(실제 data/, models/ 포함)이거나 프로젝트를 품고 있으면 거부한다."""
    root, project = Path(path).resolve(), Path(project_root).resolve()
    if root == project or project in root.parents or root in project.parents:
        raise ValueError(f"측정용 임시 루트가 프로젝트와 겹칩니다: {root}")
    return root


def resolve_report_path(text, project_root=PROJECT_ROOT):
    """보고서 경로를 프로젝트 루트 기준으로 해석한다. '-' 는 None(파일 없음). 루트 밖이면 거부."""
    if text == "-":
        return None
    root = Path(project_root).resolve()
    path = Path(text)
    path = (path if path.is_absolute() else root / path).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError(f"--report: 프로젝트 루트({root}) 안의 경로여야 합니다 (받은 값: {text!r})") from None
    return path


def check_report_target(path, overwrite):
    if path is not None and path.exists() and not overwrite:
        raise FileExistsError(f"보고서가 이미 있습니다: {path} (덮어쓰려면 --overwrite)")


def check_prerequisites(project_root=PROJECT_ROOT):
    """실행에 필요한 파일이 없으면 안내 문장 목록을 돌려준다 (비어 있으면 통과)."""
    root = Path(project_root)
    problems = []
    for name in ("model.pt", "metrics.json"):
        if not (root / "models" / "v1" / name).is_file():
            problems.append(f"models/v1/{name} 가 없습니다. 먼저 python -m scripts.train_v1 을 실행하세요")
    if not (root / "parts" / "collector" / "collector.toml").is_file():
        problems.append("parts/collector/collector.toml 이 없습니다")
    return problems


# ------------------------------------------------------------------ 항목 5: 제품 함수를 메모리에서만 감싸기
class TimerError(Exception):
    """감쌀 제품 함수를 찾지 못했다 (제품 내부 이름이 바뀜)."""


CURRENT = contextvars.ContextVar("mlops_latency_current", default=None)


def install_timers(targets, resolve):
    """targets 의 함수를 시간 기록용으로 감싼다. resolve(모듈 경로) 는 모듈을 돌려준다. (복원 함수 목록)을 돌려준다.

    하나라도 찾지 못하면 아무것도 바꾸지 않고 TimerError (찾지 못한 이름들을 알린다)."""
    found, missing = [], []
    for module_path, owner_name, attr, key in targets:
        try:
            module = resolve(module_path)
            owner = getattr(module, owner_name) if owner_name else module
            original = getattr(owner, attr)
        except (ImportError, AttributeError):
            missing.append(f"{module_path}.{owner_name + '.' if owner_name else ''}{attr}")
            continue
        found.append((owner, attr, original, key))
    if missing:
        raise TimerError("제품 내부 이름을 찾지 못했습니다: " + ", ".join(missing))
    restores = []
    for owner, attr, original, key in found:
        def make(original=original, key=key):
            def timed(*a, **k):
                t = time.perf_counter()
                try:
                    return original(*a, **k)
                finally:
                    d = CURRENT.get()
                    if d is not None:
                        d.setdefault(key, []).append((time.perf_counter() - t) * 1000)
            return timed

        setattr(owner, attr, make())
        restores.append((owner, attr, original))
    return restores


def restore_timers(restores):
    for owner, attr, original in restores:
        setattr(owner, attr, original)


# ------------------------------------------------------------------ 측정 환경 (실제 파일은 읽기만)
def pc_state():
    cmd = ("$s=(Get-Counter '\\Processor(_Total)\\% Processor Time' -SampleInterval 1 -MaxSamples 3).CounterSamples | "
           "ForEach-Object CookedValue; [math]::Round(($s | Measure-Object -Average).Average,1); "
           "(Get-Process | Where-Object { $_.ProcessName -match 'python|uvicorn' } | Measure-Object).Count; "
           "Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 8 | "
           "ForEach-Object { $_.ProcessName + ':' + [int]($_.WorkingSet64/1MB) + 'MB' }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=30).stdout.split("\n")
        out = [l.strip() for l in out if l.strip()]
    except (OSError, subprocess.SubprocessError):
        out = []
    return {"time": datetime.now().isoformat(timespec="seconds"), "cpu_pct": out[0] if out else "측정 불가",
            "python_processes": out[1] if len(out) > 1 else "?", "top_memory": out[2:]}


def real_files(project_root=PROJECT_ROOT):
    """실제 data/mlops.db(크기·수정 시각·WAL), models/v1(해시), data/pool 파일 수. 읽기만 한다."""
    root = Path(project_root)
    info = {}
    db = root / "data" / "mlops.db"
    info["mlops.db"] = [db.stat().st_size, db.stat().st_mtime_ns] if db.exists() else None
    for ext in ("-wal", "-shm"):
        p = Path(str(db) + ext)
        info["mlops.db" + ext] = p.stat().st_size if p.exists() else None
    model_dir = root / "models" / "v1"
    if model_dir.is_dir():
        for f in sorted(model_dir.iterdir()):
            info["models/v1/" + f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    pool = root / "data" / "pool"
    info["data/pool 파일 수"] = len(list(pool.iterdir())) if pool.exists() else 0
    report = root / DEFAULT_REPORT
    info[DEFAULT_REPORT] = hashlib.sha256(report.read_bytes()).hexdigest() if report.exists() else None
    return info


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_closed(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) != 0


def make_root(secret, register=True):
    """임시 폴더: 실제 v1 복사본, 수집기 설정 복사본, 임시 DB, 수집기로 만든 검사 20건. 실제 파일은 읽기만 한다."""
    from backend.database import create_db_engine, init_db, make_session_factory, session_scope
    from backend.registry import sync_models
    from backend.repository import InspectionRepository
    from parts.collector import Collector

    root = assert_safe_root(tempfile.mkdtemp(prefix="mlops_latency_"))
    shutil.copytree(PROJECT_ROOT / "models" / "v1", root / "models" / "v1")
    (root / "parts" / "collector").mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / "parts" / "collector" / "collector.toml", root / "parts" / "collector" / "collector.toml")
    (root / "data").mkdir()
    engine = create_db_engine(root / "data" / "mlops.db")
    try:
        init_db(engine)
        with session_scope(make_session_factory(engine)) as s:
            collector = Collector(root / "parts" / "collector" / "collector.toml", root=root)
            repo = InspectionRepository(s)
            for i in range(N_INSPECTIONS):
                repo.add(collector.next_sample(scenario="drift" if i % 2 else "normal", data_split="pool").inspection)
            if register:
                sync_models(s, root / "models", production="v1")
    finally:
        engine.dispose()
    return root


def register_production(root):
    from backend.database import create_db_engine, make_session_factory, session_scope
    from backend.registry import sync_models

    engine = create_db_engine(root / "data" / "mlops.db")
    try:
        with session_scope(make_session_factory(engine)) as s:
            sync_models(s, root / "models", production="v1")
    finally:
        engine.dispose()


def create_app_from_env():
    """uvicorn --factory 대상: 환경 변수의 임시 루트를 프로젝트 루트처럼 써서 앱을 만든다 (측정용 서버)."""
    from backend.main import create_app
    from backend.settings import load_settings

    return create_app(load_settings(root=Path(os.environ[ENV_ROOT]), secret_key=os.environ[ENV_SECRET]))


# ------------------------------------------------------------------ HTTP 클라이언트와 서버 프로세스
class Client:
    def __init__(self, port, secret):
        from backend.security import create_token

        self.port, self.conn = port, None
        self.headers = {"Authorization": f"Bearer {create_token('operator', secret, 480)}", "Content-Type": "application/json"}

    def call(self, method, path, body=None, fresh=False):
        """(상태, 응답 JSON, 걸린 시간 ms). 서버가 유휴 연결을 닫았으면 다시 연결한다 (연결 시간도 포함)."""
        start = time.perf_counter()
        for attempt in (0, 1):
            if self.conn is None or fresh:
                if self.conn is not None:
                    self.conn.close()
                self.conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)
                fresh = False
            try:
                self.conn.request(method, path, body=json.dumps(body) if body is not None else None, headers=self.headers)
                resp = self.conn.getresponse()
                raw = resp.read()
                break
            except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError, http.client.CannotSendRequest):
                self.conn = None
                if attempt:
                    raise
        ms = (time.perf_counter() - start) * 1000
        try:
            data = json.loads(raw) if raw else None
        except ValueError:
            data = None
        return resp.status, data, ms

    def predict(self, inspection_id, fresh=False):
        status, data, ms = self.call("POST", "/predict", {"inspection_id": inspection_id}, fresh=fresh)
        return {"status": status, "B": ms, "A": data.get("latency_ms") if status == 201 and data else None}

    def close(self):
        if self.conn is not None:
            self.conn.close()


def inspection_ids():
    return [f"INS-{i:06d}" for i in range(1, N_INSPECTIONS + 1)]


ACTIVE_SERVERS = []


class Server:
    def __init__(self, root, secret):
        self.root, self.secret, self.port, self.proc = root, secret, free_port(), None
        self.log = tempfile.NamedTemporaryFile(prefix="mlops_latency_srv_", suffix=".log", delete=False)

    def start(self):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", **{ENV_ROOT: str(self.root), ENV_SECRET: self.secret})
        self.t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "scripts.measure_latency:create_app_from_env", "--factory", "--host", "127.0.0.1",
             "--port", str(self.port), "--log-level", "warning"], cwd=str(PROJECT_ROOT), env=env, stdout=self.log, stderr=self.log)
        ACTIVE_SERVERS.append(self)
        t_up = t_loaded = None
        while time.perf_counter() - self.t0 < 90:
            try:
                c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
                c.request("GET", "/health")
                body = json.loads(c.getresponse().read())
                c.close()
                now = (time.perf_counter() - self.t0) * 1000
                if t_up is None:
                    t_up = now
                if body["model"]["loaded"]:
                    t_loaded = now
                    break
                if body["model"]["production_version"] is None:
                    break  # 운영 모델이 없으면 loaded 는 오지 않는다 (지연 로딩 시험)
            except (OSError, http.client.HTTPException, ValueError):
                pass
            time.sleep(0.02)
        return t_up, t_loaded

    def stop(self):
        pid = self.proc.pid
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
        time.sleep(0.5)
        self.log.close()
        try:
            os.unlink(self.log.name)
        except OSError:
            pass
        if self in ACTIVE_SERVERS:
            ACTIVE_SERVERS.remove(self)
        return {"port": self.port, "pid": pid, "port_closed": port_closed(self.port)}


def stop_all_servers():
    return [s.stop() for s in list(ACTIVE_SERVERS)]


# ------------------------------------------------------------------ 측정 항목
def item_startup_and_cold(root, secret, restarts, idle_seconds):
    """항목 1(시작 -> loaded) + 콜드 첫 /predict + (선택) 유휴 후 첫 /predict. 같은 루트에서 서버를 restarts 번 새로 띄운다."""
    rows, stops, ids = [], [], inspection_ids()
    idle_repeats = min(3, restarts) if idle_seconds > 0 else 0
    for i in range(restarts):
        srv = Server(root, secret)
        t_up, t_loaded = srv.start()
        cl = Client(srv.port, secret)
        cold = cl.predict(ids[0], fresh=True)
        second = cl.predict(ids[1])
        row = {"t_up_ms": t_up, "t_loaded_ms": t_loaded, "cold_B": cold["B"], "cold_A": cold["A"], "cold_status": cold["status"],
               "second_B": second["B"], "second_status": second["status"]}
        if i < idle_repeats:
            time.sleep(idle_seconds)
            idle = cl.predict(ids[2], fresh=True)  # 서버는 5초 쉬면 유휴 연결을 닫는다 -> 새 연결 포함
            row.update(idle_B=idle["B"], idle_A=idle["A"], idle_status=idle["status"])
        cl.close()
        stops.append(srv.stop())
        rows.append(row)
    return rows, stops


def item_warm_pairs_concurrent(root, secret, warm, pairs, levels):
    """같은 서버에서: 웜 연속 /predict(항목 2), generate -> predict 쌍(항목 3), 동시 요청(항목 4)."""
    srv = Server(root, secret)
    t_up, t_loaded = srv.start()
    ids = inspection_ids()
    cl = Client(srv.port, secret)
    first = cl.predict(ids[0], fresh=True)
    warm_rows = [cl.predict(ids[i % N_INSPECTIONS]) for i in range(warm)]

    pair_rows = []
    for i in range(pairs):
        t0 = time.perf_counter()
        status, data, g_ms = cl.call("POST", "/simulator/generate", {"scenario": "drift" if i % 2 else "normal"})
        row = {"gen_status": status, "gen_ms": g_ms}
        if status == 201:
            p = cl.predict(data["inspection_id"])
            row.update(pred_status=p["status"], B=p["B"], A=p["A"])
        row["pair_ms"] = (time.perf_counter() - t0) * 1000
        pair_rows.append(row)

    concurrent = {}
    for workers, each in levels:
        results, barrier = [], threading.Barrier(workers)

        def work(k):
            c = Client(srv.port, secret)
            barrier.wait()
            out = [c.predict(ids[(k * each + j) % N_INSPECTIONS]) for j in range(each)]
            c.close()
            results.extend(out)

        t0 = time.perf_counter()
        threads = [threading.Thread(target=work, args=(k,)) for k in range(workers)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        wall = time.perf_counter() - t0
        concurrent[str(workers)] = {"rows": list(results), "wall_s": wall, "rps": len(results) / wall}
    cl.close()
    return {"t_up_ms": t_up, "t_loaded_ms": t_loaded, "first": first, "warm": warm_rows, "pairs": pair_rows, "concurrent": concurrent}, srv.stop()


def item_lazy_loading(secret, repeats):
    """운영 모델 없이 서버를 띄운 뒤 production 을 등록하고 첫 /predict (지연 로딩 최악의 경우)."""
    rows, stops = [], []
    for _ in range(repeats):
        root = make_root(secret, register=False)
        try:
            srv = Server(root, secret)
            t_up, _ = srv.start()
            cl = Client(srv.port, secret)
            before = cl.predict(inspection_ids()[0], fresh=True)  # 모델 없음 -> 503
            register_production(root)
            first = cl.predict(inspection_ids()[0])
            second = cl.predict(inspection_ids()[1])
            cl.close()
            stops.append(srv.stop())
        finally:
            shutil.rmtree(root, ignore_errors=True)
        rows.append({"t_up_ms": t_up, "before_status": before["status"], "first_B": first["B"], "first_A": first["A"],
                     "first_status": first["status"], "second_B": second["B"], "second_status": second["status"]})
    return rows, stops


def item_breakdown(root, secret, warm):
    """항목 5: 같은 프로세스의 스레드 서버(실제 HTTP 소켓)에서 제품 함수를 메모리에서만 감싸 구간 시간을 잰다.
    제품 내부 이름이 바뀌어 감쌀 수 없으면 {"skipped": 이유} 를 돌려준다 (다른 항목에는 영향이 없다)."""
    import importlib

    try:
        restores = install_timers(TIMER_TARGETS, importlib.import_module)
    except TimerError as e:
        return {"skipped": str(e)}
    try:
        import uvicorn

        from backend.main import create_app
        from backend.settings import load_settings

        app = create_app(load_settings(root=root, secret_key=secret))
        records = []

        class Timed:
            def __init__(self, inner):
                self.inner = inner

            async def __call__(self, scope, receive, send):
                if scope["type"] != "http":
                    return await self.inner(scope, receive, send)
                d = {}
                tok = CURRENT.set(d)
                t0 = time.perf_counter()

                async def send2(message):
                    await send(message)
                    if message["type"] == "http.response.body" and not message.get("more_body"):
                        d["server_internal"] = [(time.perf_counter() - t0) * 1000]
                        records.append(d)

                try:
                    return await self.inner(scope, receive, send2)
                finally:
                    CURRENT.reset(tok)

        port = free_port()
        server = uvicorn.Server(uvicorn.Config(Timed(app), host="127.0.0.1", port=port, log_level="warning", lifespan="on"))
        t_start = time.perf_counter()
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        while not server.started and time.perf_counter() - t_start < 60:
            time.sleep(0.02)
        startup_ms = (time.perf_counter() - t_start) * 1000
        cl = Client(port, secret)
        ids = inspection_ids()
        cl.predict(ids[0], fresh=True)  # 콜드 한 번은 버린다
        records.clear()
        rows = [cl.predict(ids[i % N_INSPECTIONS]) for i in range(warm)]
        cl.close()
        server.should_exit = True
        thread.join(30)
        app.state.engine.dispose()
    finally:
        restore_timers(restores)
    if not records:
        return {"skipped": "구간 시간이 한 건도 기록되지 않았습니다"}
    one = lambda d, k: d[k][0] if d.get(k) else 0.0
    ens1 = [d["ensure_loaded"][0] if d.get("ensure_loaded") else 0.0 for d in records]
    ens2 = [d["ensure_loaded"][1] if len(d.get("ensure_loaded", [])) > 1 else 0.0 for d in records]
    model_compute = [one(d, "inference_predict") - e for d, e in zip(records, ens2)]
    parts = [sum(one(d, k) for k in ("auth", "get_inspection", "resolve_image", "read_image", "inference_predict", "prediction_add", "commit")) + e
             for d, e in zip(records, ens1)]
    n = min(len(records), len(rows))
    comp = {
        "auth": [one(d, "auth") for d in records], "get_inspection": [one(d, "get_inspection") for d in records],
        "resolve_image": [one(d, "resolve_image") for d in records], "ensure_loaded(요청 앞부분)": ens1,
        "read_image": [one(d, "read_image") for d in records], "model_compute(판정 안의 모델 계산)": model_compute,
        "ensure_loaded(판정 안에서 다시)": ens2, "prediction_add": [one(d, "prediction_add") for d in records],
        "commit": [one(d, "commit") for d in records],
        "framework_rest(라우팅·검증·직렬화·스레드풀)": [d["server_internal"][0] - p for d, p in zip(records, parts)],
        "network(루프백, 클라이언트-서버)": [r["B"] - d["server_internal"][0] for r, d in zip(rows[:n], records[:n])],
        "server_internal": [d["server_internal"][0] for d in records]}
    return {"startup_ms": startup_ms, "n_records": len(records), "B": [r["B"] for r in rows],
            "A": [r["A"] for r in rows if r["A"] is not None], "statuses": sorted({r["status"] for r in rows}), "components": comp}


# ------------------------------------------------------------------ 보고서 (순수 함수: 결과 dict -> 문장)
def build_report(r):
    lines = ["API 포함 판정 시간(5초 요건) 측정 보고서", "=" * 60,
             "판정 대상: (B) POST /predict 요청 전체 시간 <= 5000ms. (A) latency_ms(모델 계산)는 함께 기록.",
             "측정 방법: 임시 폴더·임시 DB·실제 v1 복사본으로 실제 uvicorn(별도 프로세스, 임시 포트)을 띄워 http.client 로 측정.",
             "항목 5(구간 분해)만 같은 프로세스의 스레드 서버(실제 HTTP 소켓)에서 제품 함수를 메모리에서만 감싸 측정.",
             f"규모: {'--quick (확인용, 판정 통계로 쓰지 않음)' if r.get('quick') else '기본'}", "",
             f"측정 {r['started']} ~ {r['finished']}", ""]
    for tag, name in (("pc_before", "측정 전"), ("pc_after", "측정 후")):
        p = r[tag]
        lines.append(f"PC 상태 {name} ({p['time']}): CPU 평균 {p['cpu_pct']}%, python 프로세스 {p['python_processes']}개, 메모리 상위: {', '.join(p['top_memory'])}")
    lines.append("")

    rows = r["startup_cold_idle"]
    up = [x["t_up_ms"] for x in rows if x["t_up_ms"]]
    loaded = [x["t_loaded_ms"] for x in rows if x["t_loaded_ms"]]
    lines += [f"[항목 1] 서버 시작 -> /health 응답 -> loaded=true ({len(rows)}회 재시작, 참고용: 요건 없음)",
              f"  첫 응답까지 ms: {fmt(summarize(up))}", f"  loaded=true 까지 ms: {fmt(summarize(loaded))}",
              "  회차별 loaded=true ms: " + ", ".join(f"{v:.0f}" for v in loaded)]
    cold = [x["cold_B"] for x in rows]
    lines += ["", f"[항목 2-콜드] 서버 시작 직후 첫 /predict (모델은 시작 때 미리 불러옴, {len(rows)}회)",
              f"  (B) {fmt(summarize(cold))}", f"  (A) latency_ms: {fmt(summarize([x['cold_A'] for x in rows if x['cold_A'] is not None]))}",
              f"  상태 코드: {sorted({x['cold_status'] for x in rows})}, 두 번째 요청 (B): {fmt(summarize([x['second_B'] for x in rows]))}"]
    idle = [x["idle_B"] for x in rows if "idle_B" in x]
    if idle:
        lines += ["", f"[추가 a] 서버 시작 후 쉰 뒤 첫 판정 (새 연결 포함, {len(idle)}회)", f"  (B) {fmt(summarize(idle))}"]

    w = r["warm_pairs_concurrent"]
    warm_B, warm_A = [x["B"] for x in w["warm"]], [x["A"] for x in w["warm"] if x["A"] is not None]
    lines += ["", f"[항목 2-웜] 한 서버에서 연속 /predict {len(warm_B)}회 (서버 시작 후 첫 요청: (B) {w['first']['B']:.1f}ms)",
              f"  (B) 전체: {fmt(summarize(warm_B))}"]
    if len(warm_B) > 10:
        lines += [f"  (B) 처음 10건: {fmt(summarize(warm_B[:10]))}", f"  (B) 11번째 이후: {fmt(summarize(warm_B[10:]))}"]
    lines += [f"  (A) latency_ms: {fmt(summarize(warm_A))}"]
    if warm_A:
        lines.append(f"  (B)-(A) 차이 평균: {statistics.fmean(b - a for b, a in zip(warm_B, warm_A)):.1f}ms")
    lines.append(f"  상태 코드: {sorted({x['status'] for x in w['warm']})}")
    pairs = w["pairs"]
    ok = [p for p in pairs if p.get("pred_status") == 201]
    lines += ["", f"[항목 3] generate -> predict 연속(시연 흐름) {len(pairs)}쌍",
              f"  generate 시간: {fmt(summarize([p['gen_ms'] for p in pairs]))}", f"  첫 generate(수집기 생성 포함): {pairs[0]['gen_ms']:.1f}ms",
              f"  predict (B): {fmt(summarize([p['B'] for p in ok]))}", f"  한 쌍 전체: {fmt(summarize([p['pair_ms'] for p in pairs]))}",
              f"  상태 코드 generate {sorted({p['gen_status'] for p in pairs})} predict {sorted({p.get('pred_status') for p in pairs})}"]
    lines += ["", "[항목 4 + 추가 b] 동시 요청 (각자 keep-alive 연결, 동시에 시작)"]
    for workers in sorted(w["concurrent"], key=int):
        c = w["concurrent"][workers]
        B = [x["B"] for x in c["rows"]]
        A = [x["A"] for x in c["rows"] if x["A"] is not None]
        lines += [f"  {workers}개 병렬: 총 {len(B)}건, 전체 {c['wall_s']:.2f}초, 처리량 {c['rps']:.1f}건/초, 상태 {sorted({x['status'] for x in c['rows']})}",
                  f"    (B) {fmt(summarize(B))}", f"    (A) {fmt(summarize(A))}"]

    lz = r["lazy"]
    lines += ["", f"[참고] 지연 로딩 첫 /predict (서버가 켜진 뒤 운영 모델을 등록한 경우, 판정 대상 아님, {len(lz)}회)",
              f"  모델 등록 전 요청 상태: {sorted({x['before_status'] for x in lz})} (503 이 정상)",
              "  첫 /predict (B) ms: " + ", ".join(f"{x['first_B']:.0f}" for x in lz) + f"  (상태 {sorted({x['first_status'] for x in lz})})",
              "  그 다음 요청 (B) ms: " + ", ".join(f"{x['second_B']:.0f}" for x in lz),
              "  첫 요청의 5000ms 대비: " + ", ".join(f"{x['first_B'] / LIMIT_MS * 100:.0f}%" for x in lz)]

    b = r["breakdown"]
    lines.append("")
    if b.get("skipped"):
        lines += [f"[항목 5] 구간 분해: 건너뜀 - {b['skipped']}", "  (제품 내부 이름이 바뀌었을 수 있습니다. scripts/measure_latency.py 의 TIMER_TARGETS 를 고치세요. 항목 1~4 판정에는 영향이 없습니다.)"]
    else:
        lines += [f"[항목 5] 구간 분해 (스레드 서버, 웜 {b['n_records']}건, 상태 {b['statuses']}; 서버 시작(선로딩 포함 lifespan) {b['startup_ms']:.0f}ms)",
                  f"  클라이언트 (B): {fmt(summarize(b['B']))}", f"  (A) latency_ms: {fmt(summarize(b['A']))}"]
        mean_B = statistics.fmean(b["B"])
        for key, vals in b["components"].items():
            s = summarize(vals)
            lines.append(f"  {key:<42} 평균 {s['mean']:7.2f}  중앙값 {s['median']:7.2f}  p95 {s['p95']:7.2f}  최댓값 {s['max']:7.2f} ms  (평균 (B)의 {s['mean'] / mean_B * 100:4.1f}%)")

    lines += ["", "서버 종료 확인(PID 로 종료, 포트 닫힘): " + ", ".join(f"포트 {s['port']} pid {s['pid']} 닫힘={s['port_closed']}" for s in r["servers"]), ""]
    same = r["real_before"] == r["real_after"]
    lines += ["실제 파일 전후 (data/mlops.db 크기·수정 시각·WAL, models/v1 해시, data/pool 파일 수, 기존 보고서 해시): " + ("동일" if same else "** 달라짐 **"),
              "  " + json.dumps(r["real_after"], ensure_ascii=False), ""]

    groups = {"콜드 첫 요청": cold, "웜 연속": warm_B, "시연 흐름 predict": [p["B"] for p in ok], "유휴 후 첫 요청": idle}
    for workers in sorted(w["concurrent"], key=int):
        groups[f"동시 {workers}개"] = [x["B"] for x in w["concurrent"][workers]["rows"]]
    worst = max(max(v) for v in groups.values() if v)
    passed = verdict(worst)
    lines += ["#### 종합 판정", f"(B) 최댓값 {worst:.1f}ms -> {'통과' if passed else '** 5000ms 초과 **'} (5000ms 대비 여유 {margin_pct(worst):.1f}%)"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ 실행
def run(settings):
    secret = secrets.token_hex(32)  # 이 실행에서만 쓰는 비밀키 (파일·저장소에 쓰지 않는다)
    result = {"started": datetime.now().isoformat(timespec="seconds"), "quick": settings["quick"]}
    result["pc_before"], result["real_before"] = pc_state(), real_files()
    roots, stops = [], []
    try:
        print("항목 1·2(콜드)·유휴 ...", flush=True)
        root = make_root(secret)
        roots.append(root)
        result["startup_cold_idle"], s = item_startup_and_cold(root, secret, settings["restarts"], settings["idle_seconds"])
        stops += s
        print("항목 2(웜)·3·4 ...", flush=True)
        root2 = make_root(secret)
        roots.append(root2)
        result["warm_pairs_concurrent"], s = item_warm_pairs_concurrent(root2, secret, settings["warm"], settings["pairs"], settings["levels"])
        stops.append(s)
        print("지연 로딩 ...", flush=True)
        result["lazy"], s = item_lazy_loading(secret, settings["lazy"])
        stops += s
        print("항목 5 구간 분해 ...", flush=True)
        root3 = make_root(secret)
        roots.append(root3)
        result["breakdown"] = item_breakdown(root3, secret, settings["warm"])
    finally:
        stops += stop_all_servers()  # 중간에 실패해도 남은 서버를 PID 로 종료한다
        for r in roots:
            shutil.rmtree(r, ignore_errors=True)
    result["servers"] = stops
    result["pc_after"], result["real_after"] = pc_state(), real_files()
    result["finished"] = datetime.now().isoformat(timespec="seconds")
    return result


def main(argv=None, project_root=PROJECT_ROOT) -> int:
    args = build_parser().parse_args(argv)
    settings = settings_from_args(args)
    try:
        report_path = resolve_report_path(args.report, project_root)
        check_report_target(report_path, args.overwrite)
    except (ValueError, FileExistsError) as e:
        print(f"실패: {e}", file=sys.stderr)
        return 1
    problems = check_prerequisites(project_root)
    if problems:
        print("실패: " + "; ".join(problems), file=sys.stderr)
        return 1
    result = run(settings)
    if result["real_before"] != result["real_after"]:
        print("경고: 실제 data/ 또는 models/ 의 파일이 측정 전후로 달라졌습니다. 보고서를 확인하세요", file=sys.stderr)
    text = build_report(result)
    if report_path is None:
        print(text)
    else:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text, encoding="utf-8")
        print(f"보고서 저장: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
