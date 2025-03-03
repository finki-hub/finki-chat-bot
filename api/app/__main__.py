from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.api.links import router as links_router
from app.api.questions import router as questions_router
from app.constants import strings
from app.data.connection import Database

schema_path = Path.cwd() / "resources" / "schema.sql"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator:
    db = Database()

    await run_migration(db)

    yield

    await db.disconnect()


async def run_migration(database: Database) -> None:
    with Path.open(schema_path) as f:
        sql = f.read()

    await database.execute(sql)


def make_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    app.include_router(questions_router, prefix="/questions")
    app.include_router(links_router, prefix="/links")

    @app.get("/", tags=["Root"])
    async def root() -> str:
        return strings.alive_response

    return app


def main() -> None:
    app = make_app()
    uvicorn.run(app, port=8880, host="0.0.0.0")  # noqa: S104


if __name__ == "__main__":
    main()
