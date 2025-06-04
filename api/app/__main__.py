from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import uvicorn
from fastapi import Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.links import router as links_router
from app.api.questions import router as questions_router
from app.data.connection import Database
from app.utils.settings import Settings

settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """App startup/shutdown: init DB and run migrations."""
    db = Database(dsn=settings.DATABASE_URL)
    app.state.db = db

    await db.init()
    await db.run_migrations()

    yield

    await db.disconnect()


def get_db(request: Request) -> Database:
    return request.app.state.db


def make_app(settings: Settings) -> FastAPI:
    app = FastAPI(
        title=settings.APP_TITLE,
        description=settings.APP_DESCRIPTION,
        version=settings.API_VERSION,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "Chat", "description": "Chat with LLMs"},
            {"name": "Questions", "description": "Manage questions"},
            {"name": "Links", "description": "Manage links"},
            {"name": "Health", "description": "Health check and API status"},
        ],
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=settings.EXPOSE_HEADERS,
    )

    db_dependency = Depends(get_db)
    common_dependencies = [db_dependency]

    app.include_router(
        questions_router,
        prefix="/questions",
        tags=["Questions"],
        dependencies=common_dependencies,
    )
    app.include_router(
        links_router,
        prefix="/links",
        tags=["Links"],
        dependencies=common_dependencies,
    )
    app.include_router(
        chat_router,
        prefix="/chat",
        tags=["Chat"],
        dependencies=common_dependencies,
    )

    @app.get("/", tags=["Health"], summary="API Status")
    async def root() -> dict[str, str]:
        return {"message": f"{settings.APP_TITLE} is running"}

    @app.get(
        "/health",
        tags=["Health"],
        summary="Application Health Check",
        response_description="The health status of the application.",
    )
    async def health_check_endpoint(
        db: Database = db_dependency,
    ) -> JSONResponse:
        db_status_message = "ok"
        db_is_healthy = True
        if not db.pool:
            db_status_message = "database_pool_not_initialized"
            db_is_healthy = False
        else:
            try:
                result = await db.fetchval("SELECT 1")
                if result != 1:
                    db_status_message = "database_query_unexpected_result"
                    db_is_healthy = False
            except Exception:
                db_status_message = "database_query_error"
                db_is_healthy = False

        app_overall_status = "ok" if db_is_healthy else "unhealthy"
        http_status_code = (
            status.HTTP_200_OK if db_is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        response_content = {
            "status": app_overall_status,
            "timestamp": datetime.now(UTC).isoformat(),
            "dependencies": {
                "database": {
                    "status": db_status_message,
                    "healthy": db_is_healthy,
                },
            },
        }
        return JSONResponse(status_code=http_status_code, content=response_content)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors(), "body": exc.body},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected internal server error occurred."},
        )

    return app


def run_application() -> None:
    application = make_app(settings)

    uvicorn.run(
        application,
        host=settings.HOST,
        port=settings.PORT,
    )


if __name__ == "__main__":
    run_application()
