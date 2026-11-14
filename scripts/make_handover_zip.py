"""조원 전달용 압축 만들기: 허용 목록의 파일만 담아 zip 하나를 만든다 (.venv, data/, models/, .git, spike/ 등이 들어가는 실수를 막는다).

실행: python -m scripts.make_handover_zip [--out 경로]

- 허용 목록(INCLUDE)에 있는 것만 담는다. 목록에 없는 폴더·파일은 자동으로 빠진다. 허용 폴더 안의 EXCLUDE_FILES 는 뺀다.
- 만든 뒤 압축 안에 금지 이름(.git, .venv, data, models, spike, 캐시)이 있으면 실패하고 만든 파일을 지운다.
- 압축 이름의 버전은 docs/data-contract.md 첫 줄에서 읽는다. 압축 안에는 최상위 폴더 mlops-mvp/ 하나가 있다.
- 기본 저장 위치는 프로젝트의 부모 폴더이고, 프로젝트 안은 거부한다. 이미 있는 파일은 덮어쓰지 않는다.
- 보내는 쪽 체크(이 도구는 하지 않는다): 압축 전에 git status 가 비어 있는지 확인한다. 작업 폴더의 현재 파일을 그대로 담는다.
"""
import argparse
import re
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOP_FOLDER = "mlops-mvp"
INCLUDE = ("docs", "common", "parts", "backend", "services", "frontend", "tests", "scripts", "CLAUDE.md", "pyproject.toml",
           "requirements.txt", ".gitignore", ".gitattributes")
EXCLUDE_FILES = ("docs/mlops-mvp-project-brief.md",)  # 허용한 폴더 안에 있어도 담지 않는 파일 (커밋하지 않기로 한 문서)
SKIP_PARTS = {"__pycache__", ".pytest_cache"}
SKIP_SUFFIXES = {".pyc"}
FORBIDDEN = {".git", ".venv", "data", "models", "spike", "__pycache__", ".pytest_cache"}  # 경로의 어느 부분이든 이 이름이면 안 된다
REQUIRED = ("docs/data-contract.md", "docs/handover.md", "requirements.txt", "pyproject.toml")
_VERSION = re.compile(r"\bv(\d+\.\d+)\b")


class HandoverError(RuntimeError):
    """압축을 만들 수 없거나, 만든 압축이 규칙에 맞지 않음."""


def contract_version(root) -> str:
    """docs/data-contract.md 첫 줄(제목)에서 버전을 읽는다. 예: '# 데이터 계약서 ... v1.1 (동결)' -> '1.1'."""
    path = Path(root) / "docs" / "data-contract.md"
    try:
        first = path.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError) as e:
        raise HandoverError(f"계약 문서의 첫 줄을 읽을 수 없습니다: {path}") from e
    m = _VERSION.search(first)
    if not m:
        raise HandoverError(f"계약 문서 첫 줄에서 버전(v1.1 같은 형식)을 찾을 수 없습니다: {first!r}")
    return m.group(1)


def default_output(root, version: str) -> Path:
    return Path(root).resolve().parent / f"mlops-mvp-handover-v{version}.zip"


def check_forbidden(names) -> None:
    for name in names:
        bad = FORBIDDEN & set(Path(name).parts)
        if bad:
            raise HandoverError(f"압축에 들어가면 안 되는 이름이 있습니다 ({', '.join(sorted(bad))}): {name}")


def collect_files(root, include=INCLUDE, exclude=EXCLUDE_FILES) -> list:
    """허용 목록의 파일(프로젝트 루트 기준 상대 경로, 정렬). 목록의 항목이 없으면 실패한다. exclude 의 파일은 뺀다."""
    root = Path(root)
    files = []
    for entry in include:
        path = root / entry
        if path.is_file():
            files.append(Path(entry))
        elif path.is_dir():
            for f in path.rglob("*"):
                rel = f.relative_to(root)
                if f.is_file() and not (SKIP_PARTS & set(rel.parts)) and f.suffix not in SKIP_SUFFIXES \
                        and rel.as_posix() not in exclude:
                    files.append(rel)
        else:
            raise HandoverError(f"허용 목록의 항목이 없습니다: {entry}")
    return sorted(files)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def build_zip(root=PROJECT_ROOT, out=None, include=INCLUDE) -> dict:
    root = Path(root)
    version = contract_version(root)
    out = Path(out) if out is not None else default_output(root, version)
    if _inside(out, root):
        raise HandoverError(f"압축을 프로젝트 안에 만들 수 없습니다 (프로젝트 밖 경로를 쓰세요): {out}")
    if out.exists():
        raise HandoverError(f"이미 있어서 덮어쓰지 않습니다: {out} (다른 --out 을 쓰거나 직접 지우세요)")

    files = collect_files(root, include)
    names = [rel.as_posix() for rel in files]
    for required in REQUIRED:
        if required not in names:
            raise HandoverError(f"필수 파일이 없습니다: {required}")
    arcnames = [f"{TOP_FOLDER}/{n}" for n in names]
    check_forbidden(arcnames)

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(out, "x", zipfile.ZIP_DEFLATED) as zf:
            for rel, arc in zip(files, arcnames):
                zf.write(root / rel, arc)
        summary = _verify(out, version, root, files)  # 만든 압축을 다시 열어 확인한다
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    return summary


def _verify(out: Path, version: str, root: Path, files: list) -> dict:
    with zipfile.ZipFile(out) as zf:
        if zf.testzip() is not None:
            raise HandoverError("압축 파일이 손상되었습니다")
        names = zf.namelist()
    check_forbidden(names)
    if {n.split("/")[0] for n in names} != {TOP_FOLDER}:
        raise HandoverError(f"압축의 최상위 폴더가 {TOP_FOLDER}/ 하나가 아닙니다")
    by_top = {}
    for rel in files:
        by_top[rel.parts[0]] = by_top.get(rel.parts[0], 0) + 1
    return {
        "version": version, "path": out, "files": len(names), "by_top": by_top,
        "zip_bytes": out.stat().st_size, "raw_bytes": sum((root / rel).stat().st_size for rel in files),
    }


def format_summary(s: dict) -> str:
    lines = [
        f"계약 버전: v{s['version']}",
        f"압축 파일: {s['path']}",
        f"압축 크기: {s['zip_bytes'] / 1024:.1f} KB (압축 전 {s['raw_bytes'] / 1024:.1f} KB)",
        f"파일 수: {s['files']}",
    ]
    lines += [f"- {top}: {n}개" for top, n in sorted(s["by_top"].items())]
    lines.append(f"제외 확인: {', '.join(sorted(FORBIDDEN))} 없음")
    return "\n".join(lines)


def main(argv=None, root=PROJECT_ROOT) -> int:
    parser = argparse.ArgumentParser(description="조원 전달용 압축(zip)을 만든다 (허용 목록의 파일만 담음)")
    parser.add_argument("--out", default=None, help="압축 저장 경로 (프로젝트 밖만 허용, 기본값: 프로젝트 부모 폴더의 mlops-mvp-handover-v<버전>.zip)")
    args = parser.parse_args(argv)
    try:
        summary = build_zip(root, args.out)
    except (HandoverError, OSError) as e:
        print(f"실패: {e}", file=sys.stderr)
        return 1
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
