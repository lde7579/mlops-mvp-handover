"""scripts/make_handover_zip.py 테스트: 임시 폴더의 가짜 프로젝트로 허용 목록·금지 이름·덮어쓰기 거부 등을 확인한다."""
import ast
import sys
import zipfile

import pytest

from scripts.make_handover_zip import (
    EXCLUDE_FILES,
    FORBIDDEN,
    PROJECT_ROOT,
    TOP_FOLDER,
    HandoverError,
    build_zip,
    check_forbidden,
    collect_files,
    contract_version,
    default_output,
    format_summary,
    main,
)

ALLOWED = {
    "docs/data-contract.md": "# 데이터 계약서 (Data Contract) v1.1 (동결)\n",
    "docs/handover.md": "# 인수 가이드\n",
    "common/schemas.py": "x = 1\n",
    "parts/collector/a.py": "x = 1\n",
    "backend/a.py": "x = 1\n",
    "services/training/a.py": "x = 1\n",
    "frontend/a.py": "x = 1\n",
    "tests/test_a.py": "x = 1\n",
    "tests/training/test_a.py": "x = 1\n",
    "scripts/s.py": "x = 1\n",
    "CLAUDE.md": "rules\n",
    "pyproject.toml": "[tool]\n",
    "requirements.txt": "numpy==1\n",
    ".gitignore": "data/\n",
    ".gitattributes": "* text=auto\n",
}
EXCLUDED = [
    ".git/config", ".venv/Lib/site.py", "data/dataset/a.png", "models/v1/model.pt", "spike/a.py",
    ".claude/settings.json", "notes.txt", "parts/collector/__pycache__/a.pyc", "tests/__pycache__/t.pyc",
    ".pytest_cache/x", "parts/collector/b.pyc", "docs/mlops-mvp-project-brief.md",
]


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    for rel, text in {**ALLOWED, **{p: "x" for p in EXCLUDED}}.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def names_of(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        return zf.namelist()


def test_zip_contains_only_allowed_files(project, tmp_path):
    out = tmp_path / "out" / "h.zip"
    summary = build_zip(project, out)

    names = names_of(out)
    assert sorted(names) == sorted(f"{TOP_FOLDER}/{p}" for p in ALLOWED)
    assert {n.split("/")[0] for n in names} == {TOP_FOLDER}
    assert all(".." not in n.split("/") and not n.startswith("/") and ":" not in n for n in names)
    for n in names:
        assert not (FORBIDDEN & set(n.split("/")))
    assert summary["files"] == len(ALLOWED) and summary["version"] == "1.1"
    with zipfile.ZipFile(out) as zf:
        assert zf.testzip() is None


def test_default_output_is_in_parent_folder_with_version(project, tmp_path):
    assert default_output(project, "1.1") == (project.resolve().parent / "mlops-mvp-handover-v1.1.zip")
    assert main([], root=project) == 0
    assert (tmp_path / "mlops-mvp-handover-v1.1.zip").is_file()


def test_existing_output_is_not_overwritten(project, tmp_path, capsys):
    out = tmp_path / "h.zip"
    out.write_bytes(b"keep")
    assert main(["--out", str(out)], root=project) == 1
    assert out.read_bytes() == b"keep"
    assert "덮어쓰지 않습니다" in capsys.readouterr().err


def test_output_inside_project_is_refused(project):
    with pytest.raises(HandoverError, match="프로젝트 안"):
        build_zip(project, project / "h.zip")
    assert not (project / "h.zip").exists()
    assert main(["--out", str(project / "sub" / "h.zip")], root=project) == 1


@pytest.mark.parametrize("missing", ["docs/handover.md", "docs/data-contract.md", "requirements.txt", "CLAUDE.md"])
def test_missing_required_or_listed_file_fails_and_leaves_no_zip(project, tmp_path, missing):
    (project / missing).unlink()
    out = tmp_path / "h.zip"
    with pytest.raises(HandoverError):
        build_zip(project, out)
    assert not out.exists()


def test_title_without_version_fails(project, tmp_path):
    (project / "docs" / "data-contract.md").write_text("# 데이터 계약서\n본문 v9.9\n", encoding="utf-8")
    with pytest.raises(HandoverError, match="버전"):
        build_zip(project, tmp_path / "h.zip")
    assert not (tmp_path / "h.zip").exists()


def test_contract_version_reads_first_line_only(project):
    assert contract_version(project) == "1.1"
    (project / "docs" / "data-contract.md").write_text("# 계약 v2.0 (동결)\n이전 v1.1\n", encoding="utf-8")
    assert contract_version(project) == "2.0"


@pytest.mark.parametrize("name", ["data", "models", ".venv", ".git", "spike"])
def test_guard_blocks_forbidden_names_even_if_listed(project, tmp_path, name):
    (project / name).mkdir(exist_ok=True)
    (project / name / "a.txt").write_text("x", encoding="utf-8")
    out = tmp_path / "h.zip"
    with pytest.raises(HandoverError, match="들어가면 안 되는 이름"):
        build_zip(project, out, include=("docs", "requirements.txt", "pyproject.toml", name))
    assert not out.exists()  # 만들던 파일도 남기지 않는다


def test_check_forbidden_looks_at_every_path_part():
    check_forbidden([f"{TOP_FOLDER}/parts/collector/a.py"])
    for bad in (f"{TOP_FOLDER}/data/a.png", f"{TOP_FOLDER}/parts/models/m.pt", f"{TOP_FOLDER}/.git/config"):
        with pytest.raises(HandoverError):
            check_forbidden([bad])


def test_collect_files_skips_caches(project):
    files = {p.as_posix() for p in collect_files(project)}
    assert files == set(ALLOWED)


def test_summary_text(project, tmp_path):
    text = format_summary(build_zip(project, tmp_path / "h.zip"))
    assert "계약 버전: v1.1" in text and "압축 크기:" in text and f"파일 수: {len(ALLOWED)}" in text
    assert "- docs: 2개" in text
    assert "제외 확인:" in text and "spike" in text and "없음" in text


def test_real_project_can_be_packaged(tmp_path):
    # 실제 프로젝트가 압축 규칙을 통과하고 (handover.md 포함), 금지 이름이 하나도 없는지 확인한다
    out = tmp_path / "real.zip"
    summary = build_zip(PROJECT_ROOT, out)
    names = names_of(out)
    assert f"{TOP_FOLDER}/docs/handover.md" in names and f"{TOP_FOLDER}/docs/data-contract.md" in names
    assert summary["version"] == contract_version(PROJECT_ROOT)
    assert default_output(PROJECT_ROOT, summary["version"]).name == f"mlops-mvp-handover-v{contract_version(PROJECT_ROOT)}.zip"
    for n in names:
        assert not (FORBIDDEN & set(n.split("/")))
    for folder in ("backend", "services", "tests/training", "tests/backend", "tests/contract"):
        assert any(n.startswith(f"{TOP_FOLDER}/{folder}/") for n in names), folder
    for excluded in EXCLUDE_FILES:
        assert f"{TOP_FOLDER}/{excluded}" not in names
    assert not any(n.endswith("secret.key") for n in names)


def test_project_brief_is_excluded_even_if_present(project):
    # 허용 폴더(docs) 안에 있어도 제외 목록의 파일은 담기지 않는다 (project 에는 이 파일이 이미 만들어져 있다)
    assert (project / "docs" / "mlops-mvp-project-brief.md").is_file()
    assert "docs/mlops-mvp-project-brief.md" not in {p.as_posix() for p in collect_files(project)}


# 외부 패키지: import 이름 -> requirements.txt 의 패키지 이름. 표에 없는 이름이 나오면 실패한다(새 의존을 알아채기 위함).
THIRD_PARTY = {
    "PIL": "pillow", "numpy": "numpy", "sklearn": "scikit-learn", "torch": "torch", "sqlalchemy": "SQLAlchemy",
    "fastapi": "fastapi", "starlette": "fastapi",  # starlette 은 fastapi 가 함께 설치한다
    "uvicorn": "uvicorn", "pydantic": "pydantic", "jwt": "PyJWT", "pytest": "pytest",
    "streamlit": "streamlit", "httpx2": "httpx2",  # 프론트(frontend/)와 그 시험
}


def test_real_zip_imports_are_all_available(tmp_path):
    # 압축 안의 모든 .py 가 import 하는 이름이 (압축 안 파일, 표준 라이브러리, 고정 표 + requirements.txt) 중 하나인지 확인한다.
    # git, docs/mlops-mvp-project-brief.md, 압축 밖 파일 없이 압축 내용만 본다.
    out = tmp_path / "real.zip"
    build_zip(PROJECT_ROOT, out)
    with zipfile.ZipFile(out) as zf:
        names = {n.removeprefix(f"{TOP_FOLDER}/") for n in zf.namelist()}
        sources = {n: zf.read(f"{TOP_FOLDER}/{n}").decode("utf-8") for n in names if n.endswith(".py")}
        requirements = zf.read(f"{TOP_FOLDER}/requirements.txt").decode("utf-8")
    required = {line.split("==")[0].strip().lower() for line in requirements.splitlines()
                if line.strip() and not line.startswith("#")}
    local_tops = {n.split("/")[0] for n in names if "/" in n}
    problems = []
    for rel, source in sources.items():
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                top = module.split(".")[0]
                if top in sys.stdlib_module_names:
                    continue
                if top in local_tops:
                    path = module.replace(".", "/")
                    if f"{path}.py" not in names and f"{path}/__init__.py" not in names:
                        problems.append(f"{rel}: 압축 안에 없는 모듈 {module}")
                elif top not in THIRD_PARTY:
                    problems.append(f"{rel}: 표에 없는 이름 {module}")
                elif THIRD_PARTY[top].lower() not in required:
                    problems.append(f"{rel}: requirements.txt 에 {THIRD_PARTY[top]} 없음")
    assert not problems, "\n".join(problems)
