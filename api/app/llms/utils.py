from app.data.connection import Database
from app.llms.ollama import generate_ollama_embeddings
from app.utils.db import embedding_to_pgvector
from app.utils.models import MODEL_COLUMNS, Model


async def fill_embeddings(model: Model, all: bool = False) -> None:
    db = Database()

    model_column = MODEL_COLUMNS[model]
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
        text_to_embed = f"Наслов: {row['name']}\nСодржина: {row['content']}"  # noqa: RUF001
        embedding = generate_embeddings(text_to_embed, model)

        await db.execute(
            f"UPDATE question SET {model_column} = $1 WHERE id = $2",  # noqa: S608
            embedding_to_pgvector(embedding),
            row["id"],
        )


def generate_embeddings(text: str, model: Model) -> list[float]:
    match model:
        case Model.LLAMA_3_3_70B:
            return generate_ollama_embeddings(text, model)
        case _:
            raise ValueError(f"Unsupported model: {model}")
