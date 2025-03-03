# mypy: disable-error-code="arg-type"

from app.data.connection import Database
from app.schema.link import CreateLinkSchema, LinkSchema, UpdateLinkSchema

db = Database()


async def get_links_query() -> list[LinkSchema]:
    query = "SELECT * FROM link ORDER BY name ASC"
    result = await db.fetch(query)

    return [LinkSchema(**row) for row in result]


async def get_link_names_query() -> list[str]:
    query = "SELECT name FROM link ORDER BY name ASC"
    result = await db.fetch(query)

    return [str(row["name"]) for row in result]


async def get_link_by_name_query(name: str) -> LinkSchema | None:
    query = "SELECT * FROM link WHERE name = $1"
    result = await db.fetchrow(query, name)

    if not result:
        return None

    return LinkSchema(**result)


async def create_link_query(
    link: CreateLinkSchema,
) -> LinkSchema | None:
    query = """
    INSERT INTO link (name, url, description, user_id)
    VALUES ($1, $2, $3, $4)
    RETURNING *
    """
    result = await db.fetchrow(
        query,
        link.name,
        link.url,
        link.description,
        link.user_id,
    )

    if not result:
        return None

    return LinkSchema(**result)


async def update_link_query(
    name: str,
    link: UpdateLinkSchema,
) -> LinkSchema | None:
    query = """
    UPDATE link
    SET name = $1, url = $2, description = $3, user_id = $4
    WHERE name = $5
    RETURNING *
    """
    result = await db.fetchrow(
        query,
        link.name,
        link.url,
        link.description,
        link.user_id,
        name,
    )

    if not result:
        return None

    return LinkSchema(**result)


async def delete_link_query(name: str) -> None:
    query = "DELETE FROM link WHERE name = $1"
    await db.execute(query, name)


async def get_nth_link_query(n: int) -> LinkSchema | None:
    query = "SELECT * FROM link ORDER BY name ASC LIMIT 1 OFFSET $1"
    result = await db.fetchrow(query, n)

    if not result:
        return None

    return LinkSchema(**result)
