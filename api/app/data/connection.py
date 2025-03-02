import os

import asyncpg

DATABASE_URL = os.getenv("DATABASE_URL")


class Database:
    _instance: "Database | None" = None

    def __new__(cls) -> "Database":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.pool = None
        return cls._instance

    async def init(self) -> None:
        """Initialize the database connection pool."""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)

    async def disconnect(self) -> None:
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def fetch(self, query: str, *args) -> list[asyncpg.Record]:
        """Fetch multiple rows from the database."""
        if not self.pool:
            await self.init()
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> asyncpg.Record | None:
        """Fetch a single row from the database."""
        if not self.pool:
            await self.init()
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def execute(self, query: str, *args) -> str:
        """Execute a query (INSERT, UPDATE, DELETE) and return status."""
        if not self.pool:
            await self.init()
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)
