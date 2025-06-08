from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

from .tools.staff import get_staff
from .utils.settings import Settings

settings = Settings()


def make_app(settings: Settings) -> FastMCP:
    mcp = FastMCP(
        title=settings.APP_TITLE,
        description=settings.APP_DESCRIPTION,
        version=settings.API_VERSION,
        port=settings.PORT,
        host=settings.HOST,
    )

    @mcp.tool(name="get_staff", description="Преземи листа од наставен кадар од ФИНКИ")
    async def get_staff_tool() -> list[TextContent]:
        result = await get_staff()

        if isinstance(result, str):
            return [TextContent(type="text", text=result)]

        return [TextContent(type="text", text=name) for name in result]

    return mcp


app = make_app(settings)
