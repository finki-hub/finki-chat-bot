import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from app.api.embeddings import router as embeddings_router
from app.api.health import router as health_router
from app.utils.settings import Settings

settings = Settings()


def make_app(settings: Settings) -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title=settings.APP_TITLE,
        description=settings.APP_DESCRIPTION,
        version=settings.API_VERSION,
        openapi_tags=[
            {"name": "Embeddings", "description": "Manage embeddings"},
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

    app.include_router(embeddings_router)
    app.include_router(health_router)

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
    """
    Run the FastAPI application using Uvicorn.
    """
    application = make_app(settings)

    uvicorn.run(
        application,
        host=settings.HOST,
        port=settings.PORT,
    )


if __name__ == "__main__":
    run_application()
