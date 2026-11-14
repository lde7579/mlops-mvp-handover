"""검사 1건을 판정해 prediction 으로 저장한다 (자동 생성이 생성 직후에 부른다: api-v1.md 4장 "자동 판정", A26).

POST /predict (backend/routers/predict.py) 와 같은 순서·같은 결과다: 모델 상태 확인 -> 이미지 읽기 -> 판정 -> 저장.
router 를 가져오지 않으려고 이미지 읽기만 여기에 한 번 더 둔다 (약 8줄, /predict 와 합치는 정리는 5주차 후보).
판정에 쓴 버전과 임계값은 InferenceService.predict 가 짝을 맞춰 돌려준다 (계약 C5).
"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

from backend.images import resolve_image
from backend.inference import InferenceService
from backend.repository import PredictionRepository
from common.schemas import Inspection, Prediction, SchemaError, validate_pil_image

log = logging.getLogger("backend")


class JudgeError(RuntimeError):
    """판정할 이미지를 읽을 수 없다 (없거나, 허용되지 않는 경로이거나, 규격이 맞지 않음). 메시지는 상태 조회의 last_error 에 쓴다."""


def _read_image(path) -> np.ndarray:
    try:
        with Image.open(path) as png:
            png.load()
            validate_pil_image(png)
            return np.asarray(png).copy()
    except (OSError, SchemaError) as e:
        log.error("판정할 이미지를 읽을 수 없습니다 (%s): %s", path, e)
        raise JudgeError("판정할 이미지를 읽을 수 없습니다") from None


def judge_inspection(session: Session, inference: InferenceService, root, inspection: Inspection,
                     now: Optional[datetime] = None) -> Prediction:
    """운영 모델로 판정해 저장한다 (커밋은 부른 쪽). 운영 모델이 없거나 불러오지 못하면 ModelUnavailableError."""
    path = resolve_image(Path(root), inspection.image_path)
    if path is None:
        raise JudgeError("이미지를 찾을 수 없습니다")
    inference.ensure_loaded(session)  # 이미지를 읽기 전에 모델 상태부터 확인한다 (/predict 와 같은 순서)
    image = _read_image(path)
    output, threshold = inference.predict(session, image)
    return PredictionRepository(session).add(
        output, inspection_id=inspection.inspection_id, threshold=threshold,
        predicted_at=datetime.now(timezone.utc) if now is None else now)
