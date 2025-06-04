# mypy: disable-error-code="arg-type"

import json

from app.data.connection import Database
from app.llms.models import MODEL_EMBEDDINGS_COLUMNS, Model
from app.schemas.questions import (
    CreateQuestionSchema,
    QuestionSchema,
    UpdateQuestionSchema,
)
from app.utils.database import embedding_to_pgvector


async def get_questions_query(db: Database) -> list[QuestionSchema]:
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


async def get_question_names_query(db: Database) -> list[str]:
    query = "SELECT name FROM question ORDER BY name ASC"
    result = await db.fetch(query)

    return [str(row["name"]) for row in result]


async def get_question_by_name_query(db: Database, name: str) -> QuestionSchema | None:
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
    db: Database,
    question: CreateQuestionSchema,
) -> QuestionSchema | None:
    query = """
    INSERT INTO question (name, content, user_id, links)
    VALUES ($1, $2, $3, $4::jsonb)
    RETURNING *
    """
    result = await db.fetchrow(
        query,
        question.name,
        question.content,
        question.user_id,
        json.dumps(question.links),
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
    db: Database,
    name: str,
    question: UpdateQuestionSchema,
) -> QuestionSchema | None:
    updates = question.model_dump(exclude_unset=True)

    query = "UPDATE question SET "

    update_values = []
    for i, (key, value) in enumerate(updates.items()):
        if key == "links":
            value = json.dumps(value)  # noqa: PLW2901
            query += f"{key} = ${i + 1}::jsonb, "
        else:
            query += f"{key} = ${i + 1}, "
        update_values.append(value)

    query += "updated_at = NOW()"
    query += f" WHERE name = ${len(updates) + 1} RETURNING *"

    result = await db.fetchrow(query, *update_values, name)

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


async def delete_question_query(db: Database, name: str) -> None:
    query = "DELETE FROM question WHERE name = $1"
    await db.execute(query, name)


async def get_nth_question_query(db: Database, n: int) -> QuestionSchema | None:
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


async def get_closest_questions(
    db: Database,
    embedded_query: list[float],
    model: Model,
    limit: int = 8,
    threshold: float = 0.5,
) -> list[QuestionSchema]:
    embedding_column = MODEL_EMBEDDINGS_COLUMNS[model]
    sql = f"""
    SELECT *, {embedding_column} <=> $1 AS distance
    FROM question
    WHERE {embedding_column} IS NOT NULL AND {embedding_column} <=> $1 < $3
    ORDER BY distance
    LIMIT $2
    """  # noqa: S608

    result = await db.fetch(
        sql,
        embedding_to_pgvector(embedded_query),
        limit,
        threshold,
    )

    return [
        QuestionSchema(
            id=row["id"],
            name=row["name"],
            content=row["content"],
            user_id=row["user_id"],
            links=json.loads(row["links"]) if row["links"] else {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            distance=row.get("distance", None),
        )
        for row in result
    ]
