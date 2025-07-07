from datetime import timedelta

from langchain_mcp_adapters.client import (  # type: ignore[import-untyped]
    Connection,
    MultiServerMCPClient,
    StreamableHttpConnection,
)

from app.utils.settings import Settings

settings = Settings()

_mcp_client: MultiServerMCPClient | None = None


def build_mcp_client() -> MultiServerMCPClient:
    """
    Return a singleton MultiServerMCPClient instance.
    If the client is not already created, it initializes a new one.
    The client is configured for all provided MCP servers.
    """
    global _mcp_client  # noqa: PLW0603

    if _mcp_client is not None:
        return _mcp_client

    connections: dict[str, Connection] = {}

    mcp_urls = settings.MCP_URLS.split(",") if settings.MCP_URLS else []

    for url in mcp_urls:
        connection: StreamableHttpConnection = {
            "url": url,
            "transport": "streamable_http",
            "headers": None,
            "timeout": timedelta(seconds=30),
            "sse_read_timeout": timedelta(seconds=60),
            "terminate_on_close": True,
            "session_kwargs": None,
            "httpx_client_factory": None,
        }
        connections[url] = connection

    print(
        f"Building MCP client with {len(connections)} connections: {list(connections.keys())}",
    )

    _mcp_client = MultiServerMCPClient(connections=connections)
    return _mcp_client
