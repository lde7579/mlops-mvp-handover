"""경계 시험: 프론트는 백엔드 API만 호출한다 (DB·모델 파일·학습 코드에 직접 접근하지 않는다, CLAUDE.md 프론트 규칙)."""
import ast
import re
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
FORBIDDEN_IMPORTS = {"backend", "common", "parts", "services", "scripts", "spike", "sqlalchemy", "sqlite3", "torch", "sklearn", "PIL"}
FORBIDDEN_CALLS = {"open"}
FORBIDDEN_PATH = re.compile(r"(?<![/\w])(data|models)/")  # "data/pool" 같은 파일 경로 (API 경로 "/models/production" 은 괜찮다)


def sources():
    files = sorted(FRONTEND.rglob("*.py"))
    assert files, "frontend 소스가 없다"
    return files


def test_no_forbidden_imports():
    for path in sources():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                assert name.split(".")[0] not in FORBIDDEN_IMPORTS, f"{path.name}: {name} 를 가져오면 안 된다"


def test_no_file_access_or_data_paths():
    for path in sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in FORBIDDEN_CALLS, f"{path.name}: {node.func.id}() 로 파일을 열면 안 된다"
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not FORBIDDEN_PATH.search(node.value), f"{path.name}: 파일 경로 같은 문자열 {node.value!r}"


def test_http_is_called_only_from_api_client():
    for path in sources():
        if path.name == "api_client.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert "httpx" not in text and "requests" not in text, f"{path.name}: HTTP 는 api_client.py 에서만 부른다"


def test_boundary_checks_really_detect_violations():
    """검사 규칙 자체가 동작하는지: 금지 예시를 넣으면 걸린다."""
    assert FORBIDDEN_PATH.search("data/pool/x.png") and FORBIDDEN_PATH.search("models/v1")
    assert not FORBIDDEN_PATH.search("/models/production") and not FORBIDDEN_PATH.search("http://127.0.0.1:8000/data")
