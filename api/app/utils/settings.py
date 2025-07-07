from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings.
    """

    APP_TITLE: str = "FINKI Chat Bot API"
    APP_DESCRIPTION: str = (
        "API for FINKI Chat Bot, managing questions, links, and LLM interactions."
    )
    API_VERSION: str = "1.0.0"

    GPU_API_URL: str = "http://gpu-api:8888"
    MCP_URLS: str = "http://local-mcp:8808/mcp"

    API_KEY: str = "your_api_key_here"
    DATABASE_URL: str = "postgresql+asyncpg://user:password@host:port/db"

    OLLAMA_URL: str = "http://ollama:11434"
    OPENAI_API_KEY: str = "your_openai_api_key_here"
    GOOGLE_API_KEY: str = "your_google_api_key_here"

    ALLOWED_ORIGINS: list[str] = ["*"]
    EXPOSE_HEADERS: list[str] = ["*"]

    HOST: str = "0.0.0.0"  # noqa: S104
    PORT: int = 8880
