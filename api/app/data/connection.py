from pathlib import Path

from asyncpg import Pool, Record, create_pool

from app.constants.db import SCHEMA_PATH
from app.utils.config import DATABASE_URL


class Database:
    _instance: "Database | None" = None

    pool: Pool[Record] | None = None

    def __new__(cls) -> "Database":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.pool = None
        return cls._instance

    async def init(self) -> None:
        """Initialize the database connection pool."""
        if self.pool is None:
            self.pool = await create_pool(
                DATABASE_URL,
                min_size=1,
                max_size=10,
            )

    async def disconnect(self) -> None:
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def fetch(self, query: str, *args: list) -> list[Record]:
        """Fetch multiple rows from the database."""
        if not self.pool:
            await self.init()
            assert self.pool is not None  # noqa: S101
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: list) -> Record | None:
        """Fetch a single row from the database."""
        if not self.pool:
            await self.init()
            assert self.pool is not None  # noqa: S101
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def execute(self, query: str, *args: list) -> str:
        """Execute a query (INSERT, UPDATE, DELETE) and return status."""
        if not self.pool:
            await self.init()
            assert self.pool is not None  # noqa: S101
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def run_migrations(self) -> None:
        """Run database migrations."""
        with Path.open(SCHEMA_PATH) as f:
            sql = f.read()

        await self.execute(sql)
