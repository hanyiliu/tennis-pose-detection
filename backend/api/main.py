from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.core.config import get_settings
from backend.api.routes.health import router as health_router
from backend.api.routes.predict import router as predict_router

settings = get_settings()

app = FastAPI(title=settings.app_name, version=settings.app_version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(predict_router)
