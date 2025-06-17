from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings.
    """

    APP_TITLE: str = "Data Usage Service"
    APP_DESCRIPTION: str = "Ingest arbitrary usage events for analytics"
    API_VERSION: str = "0.1.0"

    MONGO_URL: str = "mongodb://mongo:27017"

    ALLOWED_ORIGINS: list[str] = ["*"]
    EXPOSE_HEADERS: list[str] = ["*"]

    HOST: str = "0.0.0.0"  # noqa: S104
    PORT: int = 8088
