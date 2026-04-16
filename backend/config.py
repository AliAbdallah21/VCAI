# backend/config.py
"""
Backend configuration settings.
Loads exclusively from environment variables (via .env file or shell env).
All required secrets must be set — no insecure defaults.
"""

import os
from pydantic_settings import BaseSettings
from pydantic import field_validator
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings. Required secrets have no defaults and will raise on startup if missing."""

    # Database — required, no default
    database_url: str

    # JWT Authentication — secret required, no default
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

    # OpenRouter — required, no default
    openrouter_api_key: str

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Models
    use_mocks: bool = False
    preload_models: bool = True

    # CORS
    cors_origins: list = ["http://localhost:3000", "http://localhost:5173"]

    @field_validator("database_url")
    @classmethod
    def database_url_must_be_set(cls, v: str) -> str:
        if not v:
            raise ValueError("DATABASE_URL environment variable is required")
        return v

    @field_validator("jwt_secret")
    @classmethod
    def jwt_secret_must_be_set(cls, v: str) -> str:
        if not v:
            raise ValueError("JWT_SECRET environment variable is required")
        return v

    @field_validator("openrouter_api_key")
    @classmethod
    def openrouter_api_key_must_be_set(cls, v: str) -> str:
        if not v:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance. Raises on startup if any required variable is missing."""
    return Settings()