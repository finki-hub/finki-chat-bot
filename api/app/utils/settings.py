from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_TITLE: str = "FINKI Chat Bot API"
    APP_DESCRIPTION: str = (
        "API for FINKI Chat Bot, managing questions, links, and LLM interactions."
    )
    API_VERSION: str = "1.0.0"

    DATABASE_URL: str = "postgresql+asyncpg://user:password@host:port/db"

    ALLOWED_ORIGINS: list[str] = ["*"]
    EXPOSE_HEADERS: list[str] = ["*"]

    HOST: str = "0.0.0.0"  # noqa: S104
    PORT: int = 8880
