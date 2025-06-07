import asyncio

from app.data.connection import Database
from app.utils.settings import Settings


async def main() -> None:
    """
    Connects to the database and runs all schema migrations.
    """
    settings = Settings()
    db = Database(dsn=settings.DATABASE_URL)

    await db.init()
    await db.run_migrations()


if __name__ == "__main__":
    asyncio.run(main())
