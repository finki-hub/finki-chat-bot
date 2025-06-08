from datetime import timedelta

from langchain_mcp_adapters.client import MultiServerMCPClient, StreamableHttpConnection

from app.utils.settings import Settings

settings = Settings()

_mcp_client: MultiServerMCPClient | None = None


async def get_mcp_client() -> MultiServerMCPClient:
    """
    Return a singleton MultiServerMCPClient instance for the content API.
    If the client is not already created, it initializes a new one with the content API URL.
    """
    global _mcp_client  # noqa: PLW0603

    if _mcp_client is not None:
        return _mcp_client

    connection: StreamableHttpConnection = {
        "url": settings.CONTENT_API_URL,
        "transport": "streamable_http",
        "headers": None,
        "timeout": timedelta(seconds=30),
        "sse_read_timeout": timedelta(seconds=60),
        "terminate_on_close": True,
        "session_kwargs": None,
        "httpx_client_factory": None,
    }

    _mcp_client = MultiServerMCPClient(connections={"content-api": connection})
    return _mcp_client
