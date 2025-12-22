from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )

    PROJECT_NAME: str = "Officer Roster Optimization API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ]

    DATABASE_PATH: str = "acroster.db"

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    DEFAULT_BEAM_WIDTH: int = 20
    DEFAULT_ALPHA: float = 0.1
    DEFAULT_BETA: float = 1.0

    MAX_BEAM_WIDTH: int = 100
    MAX_CONCURRENT_GENERATIONS: int = 5


settings = Settings()
