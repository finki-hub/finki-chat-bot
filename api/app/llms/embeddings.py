from typing import overload

from app.data.connection import Database
from app.llms.models import MODEL_EMBEDDINGS_COLUMNS, Model
from app.llms.ollama import generate_ollama_embeddings
from app.llms.openai import generate_openai_embeddings
from app.utils.database import embedding_to_pgvector


@overload
async def generate_embeddings(
    text: str,
    model: Model,
) -> list[float]: ...


@overload
async def generate_embeddings(
    text: list[str],
    model: Model,
) -> list[list[float]]: ...


async def generate_embeddings(
    text: str | list[str],
    model: Model,
) -> list[float] | list[list[float]]:
    """
    Generate embeddings for the given text using the specified model.
    """
    match model:
        case Model.LLAMA_3_3_70B | Model.BGE_M3:
            return await generate_ollama_embeddings(text, model)
        case Model.TEXT_EMBEDDING_3_LARGE:
            return await generate_openai_embeddings(text, model)
        case _:
            raise ValueError(f"Unsupported model: {model}")


async def fill_embeddings(db: Database, model: Model, all: bool = False) -> None:
    """
    Fill the embeddings for all questions in the database that do not have embeddings
    for the specified model. If `all` is True, process all questions regardless of
    whether they already have embeddings.
    """
    model_column = MODEL_EMBEDDINGS_COLUMNS[model]
    rows = (
        await db.fetch(
            f"SELECT id, name, content FROM question WHERE {model_column} IS NULL",  # noqa: S608
        )
        if not all
        else await db.fetch(
            "SELECT id, name, content FROM question",
        )
    )

    for row in rows:
        text_to_embed = f"Наслов: {row['name']}\nСодржина: {row['content']}"
        embedding = await generate_embeddings(text_to_embed, model)

        await db.execute(
            f"UPDATE question SET {model_column} = $1 WHERE id = $2",  # noqa: S608
            embedding_to_pgvector(embedding),
            row["id"],
        )
