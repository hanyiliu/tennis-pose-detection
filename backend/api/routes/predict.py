from io import BytesIO

from fastapi import APIRouter, Depends, HTTPException, Request, status
from PIL import Image, UnidentifiedImageError

from backend.api.core.config import Settings, get_settings
from backend.api.schemas.prediction import PredictionResponse
from backend.api.services.model_service import ModelService, get_model_service

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    service: ModelService = Depends(get_model_service),
    settings: Settings = Depends(get_settings),
) -> PredictionResponse:
    body = await request.body()
    max_size = settings.max_upload_mb * 1024 * 1024
    if len(body) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty request body.")
    if len(body) > max_size:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Image payload too large.")

    try:
        image = Image.open(BytesIO(body)).convert("RGB")
    except UnidentifiedImageError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image payload.") from error

    try:
        prediction = service.predict(image)
    except FileNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {error}",
        ) from error

    return PredictionResponse(**prediction)
