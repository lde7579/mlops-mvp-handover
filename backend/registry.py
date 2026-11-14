"""모델 레지스트리 동기화: models/<version>/ 폴더(파일)를 model_version 테이블(DB)에 등록한다.

실행:  python -m backend.registry sync [--production v1]

- 서버가 시작할 때 자동으로 하지 않는다. "어떤 모델을 운영에 올리는가"는 사람이 정하는 결정이라 명령으로 남긴다.
- 파일은 있고 DB 에 없는 버전: candidate 로 등록한다. --production v1 을 주면 그 버전을 production 으로 등록한다
  (지금 production 이 없을 때만; 있으면 아무것도 바꾸지 않고 오류).
- DB 에는 있고 파일이 없는 버전: 지우지 않고 알린다 (그 버전이 production 이면 /health 의 loaded 는 false, /predict 는 503).
- 둘 다 있는데 지표가 다른 버전: 덮어쓰지 않고 알린다.
- metrics.json 이 깨졌거나 폴더 이름이 v숫자가 아니면 건너뛰고 알린다. 같은 명령을 여러 번 해도 안전하다.
- 파일 위치는 parts/predictor 설정(predictor.toml)의 models_dir 과 같은 곳을 본다 (Predictor 가 읽는 곳과 같아야 한다).
"""
import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from backend.database import create_db_engine, init_db, make_session_factory, session_scope
from backend.repository import ModelVersionRepository, Stage
from backend.settings import PROJECT_ROOT, Settings, load_settings
from common.schemas import ModelMetrics

VERSION_NAME = re.compile(r"^v[0-9]+$")


class RegistryError(Exception):
    """동기화를 시작할 수 없다 (아무것도 바꾸지 않았다)."""


@dataclass
class SyncReport:
    registered: list = field(default_factory=list)    # [(version, stage)]
    unchanged: list = field(default_factory=list)     # [version]
    missing_files: list = field(default_factory=list)  # [version]  DB 에는 있는데 파일이 없음
    mismatched: list = field(default_factory=list)    # [version]  DB 와 파일의 지표가 다름
    skipped: list = field(default_factory=list)       # [(폴더 이름, 이유)]

    def lines(self) -> list:
        out = [f"새로 등록: {', '.join(f'{v}({s})' for v, s in self.registered) or '없음'}",
               f"이미 등록됨(변경 없음): {', '.join(self.unchanged) or '없음'}"]
        if self.missing_files:
            out.append(f"주의 - DB 에는 있는데 파일이 없음: {', '.join(self.missing_files)}")
        if self.mismatched:
            out.append(f"주의 - DB 와 파일의 지표가 다름(덮어쓰지 않음): {', '.join(self.mismatched)}")
        out += [f"건너뜀 - {name}: {reason}" for name, reason in self.skipped]
        return out


def models_root(root=PROJECT_ROOT) -> Path:
    """모델 폴더. predictor.toml 의 models_dir 을 Predictor 와 같은 방식으로(프로젝트 루트 기준) 해석한다."""
    from parts.predictor.config import load_config  # torch 를 가져오지 않는 설정 읽기

    return Path(root) / load_config().models_dir


def _files_present(folder: Path) -> bool:
    return (folder / "model.pt").is_file() and (folder / "metrics.json").is_file()


def _read_metrics(folder: Path, version: str) -> ModelMetrics:
    metrics = ModelMetrics.from_dict(json.loads((folder / "metrics.json").read_text(encoding="utf-8")))
    if metrics.version != version:
        raise ValueError(f"metrics.json 의 version({metrics.version})이 폴더 이름({version})과 다릅니다")
    return metrics


def _number(version: str) -> int:
    return int(version[1:])


def sync_models(session, models_dir, *, production: Optional[str] = None) -> SyncReport:
    """models_dir 아래 버전 폴더를 DB 와 맞춘다. 커밋은 호출하는 쪽(session_scope)이 한다."""
    models_dir = Path(models_dir)
    repo = ModelVersionRepository(session)
    report = SyncReport()

    found: dict[str, ModelMetrics] = {}
    folders = sorted((p for p in models_dir.iterdir() if p.is_dir()), key=lambda p: p.name) if models_dir.is_dir() else []
    for folder in folders:
        if not VERSION_NAME.match(folder.name):
            report.skipped.append((folder.name, "버전 폴더 이름이 아님 (v1, v2 형태만 대상)"))
            continue
        if not _files_present(folder):
            report.skipped.append((folder.name, "model.pt 또는 metrics.json 이 없음"))
            continue
        try:
            found[folder.name] = _read_metrics(folder, folder.name)
        except Exception as e:  # 깨진 JSON, 스키마 위반, 버전 불일치
            report.skipped.append((folder.name, f"metrics.json 을 읽을 수 없음: {e}"))

    if production is not None:
        if production not in found:
            raise RegistryError(f"--production {production}: 등록할 수 있는 모델 파일이 없습니다")
        existing = repo.get(production)
        current = repo.get_production()
        if existing is not None and existing.stage is not Stage.PRODUCTION:
            raise RegistryError(f"{production} 은 이미 {existing.stage.value} 로 등록되어 있습니다 "
                                "(운영 모델 변경은 승인 절차로 합니다)")
        if current is not None and current.version != production:
            raise RegistryError(f"이미 운영 모델({current.version})이 있습니다. 변경하지 않았습니다")

    for version in sorted(found, key=_number):
        record = repo.get(version)
        if record is None:
            stage = Stage.PRODUCTION if version == production else Stage.CANDIDATE
            repo.register(found[version], stage=stage)
            report.registered.append((version, stage.value))
        else:
            report.unchanged.append(version)
            if record.metrics.to_dict() != found[version].to_dict():
                report.mismatched.append(version)

    for record in repo.list():
        if not _files_present(models_dir / record.version):
            report.missing_files.append(record.version)
    return report


def main(argv=None, *, settings: Optional[Settings] = None, out=print) -> int:
    parser = argparse.ArgumentParser(prog="python -m backend.registry", description="models/ 폴더의 모델을 DB 에 등록한다")
    sub = parser.add_subparsers(dest="command", required=True)
    sync = sub.add_parser("sync", help="파일이 있고 DB 에 없는 모델을 등록한다")
    sync.add_argument("--production", metavar="VERSION", help="운영 모델이 아직 없을 때, 이 버전을 운영(production)으로 등록한다 (예: v1)")
    args = parser.parse_args(argv)

    # 비밀키는 쓰지 않는다 (data/secret.key 를 만들지 않도록 더미 값을 준다)
    settings = settings if settings is not None else load_settings(secret_key="x" * 32)
    engine = create_db_engine(settings.db_path)
    try:
        init_db(engine)
        with session_scope(make_session_factory(engine)) as session:
            report = sync_models(session, models_root(settings.root), production=args.production)
    except RegistryError as e:
        print(f"실패: {e}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()
    for line in report.lines():
        out(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
