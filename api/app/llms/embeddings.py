import json
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import overload

from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse

from app.data.connection import Database
from app.llms.google import generate_google_embeddings
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

        case Model.TEXT_EMBEDDING_004:
            return await generate_google_embeddings(text, model)

        case _:
            raise ValueError(f"Unsupported model: {model}")


async def stream_fill_embeddings(
    db: Database,
    model: Model,
    *,
    questions: list[str] | None = None,
    all: bool = False,
) -> StreamingResponse:
    """
    Stream progress of filling embeddings for all (or missing) questions.
    Emits one SSE event per question as JSON.
    """
    model_column = MODEL_EMBEDDINGS_COLUMNS.get(model)
    if model_column is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model: {model}",
        )

    if all:
        rows = await db.fetch("SELECT id, name, content FROM question")
    elif questions:
        placeholders = ",".join(["$" + str(i + 1) for i in range(len(questions))])
        rows = await db.fetch(
            f"SELECT id, name, content FROM question WHERE name IN ({placeholders})",  # noqa: S608
            *questions,
        )
    else:
        rows = await db.fetch(
            f"SELECT id, name, content FROM question WHERE {model_column} IS NULL",  # noqa: S608
        )
    total = len(rows)

    async def _gen() -> AsyncGenerator[str]:
        for idx, row in enumerate(rows, start=1):
            qid = row["id"]
            name = row["name"]
            text = f"Наслов: {name}\nСодржина: {row['content']}"
            try:
                embedding = await generate_embeddings(text, model)
                await db.execute(
                    f"UPDATE question SET {model_column} = $1 WHERE id = $2",  # noqa: S608
                    embedding_to_pgvector(embedding),
                    qid,
                )
                result = "ok"
            except Exception as e:
                result = f"error: {e!r}"

            payload = {
                "status": result,
                "index": idx,
                "total": total,
                "id": str(qid),
                "name": name,
                "ts": datetime.now(UTC).isoformat() + "Z",
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
    )
