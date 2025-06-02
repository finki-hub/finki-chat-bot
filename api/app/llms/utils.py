from app.data.connection import Database
from app.llms.ollama import generate_ollama_embeddings
from app.llms.prompts import SYSTEM_PROMPT
from app.schema.question import QuestionSchema
from app.utils.database import embedding_to_pgvector
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
        text_to_embed = f"Наслов: {row['name']}\nСодржина: {row['content']}"
        embedding = await generate_embeddings(text_to_embed, model)

        await db.execute(
            f"UPDATE question SET {model_column} = $1 WHERE id = $2",  # noqa: S608
            embedding_to_pgvector(embedding),
            row["id"],
        )


async def generate_embeddings(text: str, model: Model) -> list[float]:
    match model:
        case Model.LLAMA_3_3_70B:
            return await generate_ollama_embeddings(text, model)
        case _:
            raise ValueError(f"Unsupported model: {model}")


def build_prompt(context: str, text: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nКонтекст:\n{context}\n\nПрашање: {text}\n\nОдговор:"


def build_context(questions: list[QuestionSchema]) -> str:
    return "\n".join(
        [f"Наслов: {q.name}\nСодржина: {q.content}" for q in questions],
    )
