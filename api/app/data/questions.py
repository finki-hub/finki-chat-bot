# mypy: disable-error-code="arg-type"

import json

from app.data.connection import Database
from app.schema.question import (
    CreateQuestionSchema,
    QuestionSchema,
    UpdateQuestionSchema,
)

db = Database()


async def get_questions_query() -> list[QuestionSchema]:
    query = "SELECT * FROM question ORDER BY name ASC"
    result = await db.fetch(query)

    return [
        QuestionSchema(
            id=row["id"],
            name=row["name"],
            content=row["content"],
            user_id=row["user_id"],
            links=json.loads(row["links"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        for row in result
    ]


async def get_question_names_query() -> list[str]:
    query = "SELECT name FROM question ORDER BY name ASC"
    result = await db.fetch(query)

    return [str(row["name"]) for row in result]


async def get_question_by_name_query(name: str) -> QuestionSchema | None:
    query = "SELECT * FROM question WHERE name = $1"
    result = await db.fetchrow(query, name)

    if not result:
        return None

    return QuestionSchema(
        id=result["id"],
        name=result["name"],
        content=result["content"],
        user_id=result["user_id"],
        links=json.loads(result["links"]),
        created_at=result["created_at"],
        updated_at=result["updated_at"],
    )


async def create_question_query(
    question: CreateQuestionSchema,
) -> QuestionSchema | None:
    query = """
    INSERT INTO question (name, content, user_id, links)
    VALUES ($1, $2, $3, $4)
    RETURNING *
    """
    result = await db.fetchrow(
        query,
        question.name,
        question.content,
        question.user_id,
        question.links,
    )

    if not result:
        return None

    return QuestionSchema(
        id=result["id"],
        name=result["name"],
        content=result["content"],
        user_id=result["user_id"],
        links=json.loads(result["links"]),
        created_at=result["created_at"],
        updated_at=result["updated_at"],
    )


async def update_question_query(
    name: str,
    question: UpdateQuestionSchema,
) -> QuestionSchema | None:
    updates = question.model_dump(exclude_unset=True)

    query = "UPDATE question SET "

    for i, key in enumerate(updates.keys()):
        query += f"{key} = ${i + 1}, "

    query += "updated_at = NOW()"
    query += f" WHERE name = ${i + 1} RETURNING *"

    result = await db.fetchrow(query, *updates.values(), name)

    if not result:
        return None

    return QuestionSchema(
        id=result["id"],
        name=result["name"],
        content=result["content"],
        user_id=result["user_id"],
        links=json.loads(result["links"]),
        created_at=result["created_at"],
        updated_at=result["updated_at"],
    )


async def delete_question_query(name: str) -> None:
    query = "DELETE FROM question WHERE name = $1"
    await db.execute(query, name)


async def get_nth_question_query(n: int) -> QuestionSchema | None:
    query = "SELECT * FROM question OFFSET $1 LIMIT 1 ORDER BY name ASC"
    result = await db.fetchrow(query, n)

    if result is None:
        return None

    return QuestionSchema(
        id=result["id"],
        name=result["name"],
        content=result["content"],
        user_id=result["user_id"],
        links=json.loads(result["links"]),
        created_at=result["created_at"],
        updated_at=result["updated_at"],
    )
