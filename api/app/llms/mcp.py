from langchain_mcp_adapters.client import (  # type: ignore[import-untyped]
    Connection,
    MultiServerMCPClient,
    SSEConnection,
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

    mcp_http_urls = settings.MCP_HTTP_URLS.split(",") if settings.MCP_HTTP_URLS else []
    mcp_sse_urls = settings.MCP_SSE_URLS.split(",") if settings.MCP_SSE_URLS else []

    for url in mcp_http_urls:
        print(f"Adding Streamable HTTP connection to MCP client: {url}")
        streamable_connection: StreamableHttpConnection = {
            "url": url,
            "transport": "streamable_http",
        }
        connections[url] = streamable_connection

    for url in mcp_sse_urls:
        print(f"Adding SSE connection to MCP client: {url}")
        sse_connection: SSEConnection = {
            "url": url,
            "transport": "sse",
        }
        connections[url] = sse_connection

    print(
        f"Building MCP client with {len(connections)} connections: {list(connections.keys())}",
    )

    _mcp_client = MultiServerMCPClient(connections=connections)

    return _mcp_client
