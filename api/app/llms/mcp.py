import logging

from langchain_mcp_adapters.client import (  # type: ignore[import-untyped]
    Connection,
    MultiServerMCPClient,
    SSEConnection,
    StreamableHttpConnection,
)

from app.utils.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

mcp_client: MultiServerMCPClient | None = None


def build_mcp_client() -> MultiServerMCPClient:
    """
    Return a singleton MultiServerMCPClient instance.
    If the client is not already created, it initializes a new one.
    The client is configured for all provided MCP servers.
    """
    global mcp_client  # noqa: PLW0603

    logger.info("Building MCP client...")

    if mcp_client is not None:
        return mcp_client

    connections: dict[str, Connection] = {}

    mcp_http_urls = settings.MCP_HTTP_URLS.split(",") if settings.MCP_HTTP_URLS else []
    mcp_sse_urls = settings.MCP_SSE_URLS.split(",") if settings.MCP_SSE_URLS else []

    for url in mcp_http_urls:
        logger.info(
            "Adding streamable HTTP connection to MCP client: %s",
            url,
        )

        streamable_connection: StreamableHttpConnection = {
            "url": url,
            "transport": "streamable_http",
        }
        connections[url] = streamable_connection

    for url in mcp_sse_urls:
        logger.info(
            "Adding SSE connection to MCP client: %s",
            url,
        )

        sse_connection: SSEConnection = {
            "url": url,
            "transport": "sse",
        }
        connections[url] = sse_connection

    logger.info(
        "Building MCP client with %d connections: %s",
        len(connections),
        list(connections.keys()),
    )

    mcp_client = MultiServerMCPClient(connections=connections)

    return mcp_client
