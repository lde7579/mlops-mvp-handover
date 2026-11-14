"""POST /predict: 검사 1건을 판정해 prediction 으로 저장한다 (운영자 이상)."""
import logging
from datetime import datetime, timezone
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends
from PIL import Image

from backend.api_models import ErrorResponse, PredictionOut, PredictRequest
from backend.deps import InferenceDep, SessionDep, SettingsDep, User, require_role
from backend.errors import ApiError
from backend.images import resolve_image
from backend.inference import ModelUnavailableError
from backend.repository import InspectionRepository, PredictionRepository
from backend.security import Role
from common.schemas import SchemaError, validate_pil_image

router = APIRouter(tags=["predict"])
log = logging.getLogger("backend")

_ERRORS = {code: {"model": ErrorResponse} for code in (400, 401, 403, 404, 500, 503)}


def read_image(path) -> np.ndarray:
    """PNG 를 uint8 (128, 128) 배열로 읽는다. 규격이 맞지 않거나 열 수 없으면 ApiError(500): 저장된 데이터가 이상한 것이다."""
    try:
        with Image.open(path) as png:
            png.load()
            validate_pil_image(png)
            return np.asarray(png).copy()
    except (OSError, SchemaError) as e:
        log.error("판정할 이미지를 읽을 수 없습니다 (%s): %s", path, e)
        raise ApiError(500) from None


@router.post("/predict", status_code=201, response_model=PredictionOut, responses=_ERRORS)
def predict(body: PredictRequest, session: SessionDep, settings: SettingsDep, inference: InferenceDep,
            _: Annotated[User, Depends(require_role(Role.OPERATOR))]) -> PredictionOut:
    """inspection_id 의 이미지를 운영 모델로 판정하고 prediction 으로 저장해 돌려준다. 같은 검사를 여러 번 판정해도 된다."""
    inspection = InspectionRepository(session).get(body.inspection_id)
    if inspection is None:
        raise ApiError(404, "검사를 찾을 수 없습니다")
    path = resolve_image(settings.root, inspection.image_path)
    if path is None:
        raise ApiError(404, "이미지를 찾을 수 없습니다")
    try:
        inference.ensure_loaded(session)  # 이미지를 읽기 전에 모델 상태부터 확인한다
        image = read_image(path)
        output, threshold = inference.predict(session, image)
    except ModelUnavailableError as e:
        raise ApiError(503, str(e)) from None
    prediction = PredictionRepository(session).add(
        output, inspection_id=inspection.inspection_id, threshold=threshold, predicted_at=datetime.now(timezone.utc))
    return PredictionOut.of(prediction)
