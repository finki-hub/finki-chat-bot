import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from api.embeddings import router as embeddings_router
from app.constants import strings


def make_app() -> FastAPI:
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    app.include_router(embeddings_router, prefix="/")

    @app.get("/", tags=["Root"])
    async def root() -> str:
        return strings.API_RUNNING

    return app


def main() -> None:
    app = make_app()
    uvicorn.run(app, port=8888, host="0.0.0.0")  # noqa: S104


if __name__ == "__main__":
    main()
