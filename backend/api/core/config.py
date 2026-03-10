from functools import lru_cache
from pathlib import Path

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="backend/.env", env_file_encoding="utf-8", case_sensitive=False)

    app_name: str = "Tennis Pose Detection API"
    app_version: str = "0.1.0"

    bbox_model_path: str = "exports/bbox_best.pt"
    keypoint_model_path: str = "exports/keypoint_best_state_dict.pt"
    pose_model_path: str = "exports/colab_e2e_best_2.pt"

    device: str = _default_device()
    max_upload_mb: int = 10
    cors_origins: str = "http://localhost:4200"

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    @property
    def cors_origins_list(self) -> list[str]:
        return [part.strip() for part in self.cors_origins.split(",") if part.strip()]

    def resolve_path(self, relative_or_abs_path: str) -> Path:
        path = Path(relative_or_abs_path)
        if path.is_absolute():
            return path
        return self.project_root / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
