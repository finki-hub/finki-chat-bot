import logging
from pathlib import Path

from asyncpg import Pool, Record, create_pool

from app.constants.db import SCHEMA_PATH
from app.utils.config import DATABASE_URL

db_logger = logging.getLogger(__name__)


class Database:
    """
    A singleton class to manage asynchronous database operations using asyncpg.

    This class handles connection pooling, executing queries, and running
    database migrations. It ensures that only one instance of the database
    manager exists throughout the application.
    """

    _instance: "Database | None" = None
    pool: Pool | None = None

    def __new__(cls) -> "Database":
        """
        Create a new Database instance or return the existing one (Singleton).

        Returns:
            Database: The singleton instance of the Database class.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.pool = None
        return cls._instance

    async def init(self) -> None:
        """
        Initialize the database connection pool.

        This method creates an asyncpg connection pool using the DATABASE_URL
        setting. It should be called once during application startup.
        If the pool is already initialized, this method does nothing.

        Raises:
            Exception: If there is an error during pool creation (e.g.,
                       cannot connect to the database).
        """
        if self.pool is None:
            try:
                db_logger.info("Initializing database pool for %s", DATABASE_URL)
                self.pool = await create_pool(
                    dsn=DATABASE_URL,
                    min_size=1,
                    max_size=10,
                )
                db_logger.info("Database pool initialized successfully.")
            except Exception:
                db_logger.exception("Failed to initialize database pool:")
                raise
        else:
            db_logger.info("Database pool already initialized.")

    async def disconnect(self) -> None:
        """
        Close the database connection pool.

        This method should be called during application shutdown to gracefully
        close all database connections.
        """
        if self.pool:
            db_logger.info("Closing database connection pool.")
            await self.pool.close()
            self.pool = None
            db_logger.info("Database connection pool closed.")

    async def _ensure_pool(self) -> Pool:
        """
        Ensure the connection pool is initialized and return it.

        If the pool is not initialized, this method will attempt to initialize it.
        This is primarily a helper method for other data access methods.

        Returns:
            Pool: The initialized asyncpg connection pool.

        Raises:
            RuntimeError: If the pool remains None after an initialization attempt,
                          indicating a critical failure.
        """
        if self.pool is None:
            db_logger.warning("Pool not initialized. Attempting to initialize now.")
            await self.init()

        if self.pool is None:
            err_msg = "Database pool is None after explicit initialization attempt."
            db_logger.error(err_msg)
            raise RuntimeError(err_msg)
        return self.pool

    async def fetch(self, query: str, *args: object) -> list[Record]:
        """
        Execute a SQL query and fetch all resulting rows.

        Args:
            query (str): The SQL query string to execute.
                         It can contain placeholders (e.g., $1, $2) for parameters.
            *args (object): Positional arguments to bind to the query's placeholders.

        Returns:
            list[Record]: A list of asyncpg.Record objects representing the rows.
                          Returns an empty list if the query yields no rows.

        Raises:
            RuntimeError: If the database pool cannot be ensured.
            asyncpg.exceptions.PostgresError: For errors during query execution.
        """
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: object) -> Record | None:
        """
        Execute a SQL query and fetch the first resulting row.

        Args:
            query (str): The SQL query string to execute.
                         It can contain placeholders (e.g., $1, $2) for parameters.
            *args (object): Positional arguments to bind to the query's placeholders.

        Returns:
            Record | None: An asyncpg.Record object for the first row,
                           or None if the query yields no rows.

        Raises:
            RuntimeError: If the database pool cannot be ensured.
            asyncpg.exceptions.PostgresError: For errors during query execution.
        """
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(
        self,
        query: str,
        *args: object,
        column: int = 0,
    ) -> object:
        """
        Execute a SQL query and fetch a single value from the first row.

        This is useful for queries that are expected to return a single
        column from a single row (e.g., SELECT COUNT(*)..., SELECT id...).

        Args:
            query (str): The SQL query string to execute.
                         It can contain placeholders (e.g., $1, $2) for parameters.
            *args (object): Positional arguments to bind to the query's placeholders.
            column (int, optional): The 0-based index of the column to retrieve
                                    from the result row. Defaults to 0.

        Returns:
            object: The value from the specified column of the first row,
                    or None if the query yields no rows or the column is NULL.
                    The type of the returned value depends on the database column type.

        Raises:
            RuntimeError: If the database pool cannot be ensured.
            asyncpg.exceptions.PostgresError: For errors during query execution.
        """
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval(query, *args, column=column)

    async def execute(self, query: str, *args: object) -> str:
        """
        Execute a SQL command (e.g., INSERT, UPDATE, DELETE, DDL statements).

        Args:
            query (str): The SQL command string to execute.
                         It can contain placeholders (e.g., $1, $2) for parameters.
            *args (object): Positional arguments to bind to the command's placeholders.

        Returns:
            str: The status string returned by asyncpg, typically indicating
                 the command tag (e.g., "INSERT 0 1", "UPDATE 5", "CREATE TABLE").

        Raises:
            RuntimeError: If the database pool cannot be ensured.
            asyncpg.exceptions.PostgresError: For errors during command execution.
        """
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def run_migrations(self) -> None:
        """
        Run database migrations by executing SQL from a schema file.

        Reads SQL commands from the file specified by `SCHEMA_PATH` and
        executes them against the database. This is typically used for
        setting up or updating the database schema.

        Raises:
            RuntimeError: If the database pool cannot be ensured.
            FileNotFoundError: If the schema file at `SCHEMA_PATH` is not found.
            Exception: For errors reading the schema file or executing migrations.
        """
        await self._ensure_pool()
        db_logger.info("Reading schema from %s", SCHEMA_PATH)
        try:
            with Path.open(SCHEMA_PATH, "r") as f:
                sql = f.read()
        except FileNotFoundError:
            db_logger.exception("Schema file not found at %s:", SCHEMA_PATH)
            raise
        except Exception:
            db_logger.exception("Error reading schema file %s:", SCHEMA_PATH)
            raise

        if not sql.strip():
            db_logger.warning(
                "Schema file %s is empty. No migrations to run.",
                SCHEMA_PATH,
            )
            return

        db_logger.info("Executing migrations...")
        try:
            await self.execute(sql)
            db_logger.info("Migrations executed successfully.")
        except Exception:
            db_logger.exception("Error executing migrations:")
            raise
