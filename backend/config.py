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

    # LLM backend selection
    use_openrouter: bool = False

    # Models
    use_mocks: bool = False
    preload_models: bool = True

    # CORS
    cors_origins: list = ["http://localhost:3000", "http://localhost:5173", "https://app.vcai.com"]

    # Multi-tenancy / SaaS (safe defaults, no new required secrets)
    trial_days: int = 14
    frontend_base_url: str = "http://localhost:5173"

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
    """
    Get cached settings instance.

    On a missing required variable, raises RuntimeError with a clean,
    actionable message instead of leaking Pydantic's verbose ValidationError
    traceback. The result is cached, so this runs at most once per process.
    """
    from pydantic import ValidationError

    try:
        return Settings()
    except ValidationError as exc:
        # Extract the missing-field names from Pydantic's error structure
        missing = []
        for err in exc.errors():
            if err.get("type") in ("missing", "value_error"):
                # err["loc"] is a tuple like ("database_url",)
                if err.get("loc"):
                    missing.append(str(err["loc"][0]).upper())
        if not missing:
            raise

        bullet = "\n  - ".join(sorted(set(missing)))
        # ASCII-only so the message prints cleanly on Windows consoles too.
        line = "=" * 67
        msg = (
            "\n"
            f"{line}\n"
            "  VCAI startup failed: required environment variables missing\n"
            f"{line}\n"
            f"  Missing or empty:\n  - {bullet}\n\n"
            "  Fix this by either:\n"
            "    1. Adding the variables to your .env file at the project root, or\n"
            "    2. Exporting them in your shell before running the server.\n\n"
            "  Example .env:\n"
            "    DATABASE_URL=postgresql://user:pass@localhost:5432/vcai\n"
            "    JWT_SECRET=<a long random string>\n"
            "    OPENROUTER_API_KEY=sk-or-...\n"
            f"{line}\n"
        )
        raise RuntimeError(msg) from None