"""모델 저장·불러오기: models/<version>/model.pt + metrics.json (docs/data-contract.md 4장)."""
import json
import re
import shutil
import uuid
from pathlib import Path

import torch

from common.schemas import ModelMetrics

_VERSION = re.compile(r"^v[0-9]+$")


class ModelLoadError(RuntimeError):
    """모델 파일을 불러오지 못함. 기존에 올려 둔 모델은 그대로 유지된다."""


def check_version(version) -> str:
    """버전 이름은 v1, v2 ... 형태만 허용한다 (폴더 이름으로 쓰이므로 경로 문자도 막는다)."""
    if not isinstance(version, str) or not _VERSION.match(version):
        raise ValueError(f"version: v1, v2 처럼 'v' 다음에 숫자여야 합니다 (받은 값: {version!r})")
    return version


def version_dir(models_root: Path, version: str) -> Path:
    return Path(models_root) / check_version(version)


def save_version(models_root: Path, version: str, state_dict: dict, metrics: ModelMetrics) -> Path:
    """임시 폴더에 다 쓴 뒤 한 번에 옮긴다. 도중에 실패하면 반쪽짜리 폴더를 남기지 않는다."""
    final = version_dir(models_root, version)
    if final.exists():
        raise FileExistsError(f"이미 있는 버전입니다: {version}")
    tmp = Path(models_root) / f".tmp-{version}-{uuid.uuid4().hex}"
    tmp.mkdir(parents=True)
    try:
        torch.save(state_dict, tmp / "model.pt")
        (tmp / "metrics.json").write_text(
            json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.rename(final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


def load_version(models_root: Path, version: str):
    """(가중치, ModelMetrics) 를 돌려준다. 문제가 있으면 ModelLoadError."""
    folder = version_dir(models_root, version)
    try:
        metrics = ModelMetrics.from_dict(json.loads((folder / "metrics.json").read_text(encoding="utf-8")))
        if metrics.version != version:
            raise ValueError(f"metrics.json 의 version({metrics.version})이 폴더 이름({version})과 다릅니다")
        state = torch.load(folder / "model.pt", weights_only=True, map_location="cpu")
    except Exception as e:
        raise ModelLoadError(f"모델 {version} 을 불러오지 못했습니다: {e}") from e
    return state, metrics
