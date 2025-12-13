"""Application configuration using Pydantic settings."""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "SAM3 Video Segmentation API"
    app_version: str = "0.1.0"
    debug: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS - allow all localhost/127.0.0.1 origins for development
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173", "http://127.0.0.1:55392", "*"]

    # Supabase
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Modal
    modal_token_id: Optional[str] = None
    modal_token_secret: Optional[str] = None

    # API (for callbacks)
    api_base_url: str = "http://localhost:8000"

    # Storage
    upload_bucket: str = "uploads"
    output_bucket: str = "outputs"
    max_video_size_mb: int = 100
    max_video_duration_seconds: int = 60
    allowed_video_extensions: list[str] = [".mp4", ".mov", ".avi", ".mkv", ".webm"]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
