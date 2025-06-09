import json
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import overload

from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse

from app.data.connection import Database
from app.llms.google import generate_google_embeddings
from app.llms.gpu_api import generate_gpu_api_embeddings
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

        case Model.MULTILINGUAL_E5_LARGE:
            return await generate_gpu_api_embeddings(text, model)

        case _:
            raise ValueError(f"Unsupported model: {model}")


async def stream_fill_embeddings(
    db: Database,
    model: Model,
    *,
    questions: list[str] | None = None,
    all_questions: bool = False,
    all_models: bool = False,
) -> StreamingResponse:
    """
    Stream progress of filling embeddings for questions.
    Can process a single model or all available embedding models.
    Emits one SSE event per question-model combination as JSON.
    """
    models_to_process: list[Model]
    if all_models:
        models_to_process = list(MODEL_EMBEDDINGS_COLUMNS.keys())
    else:
        if model not in MODEL_EMBEDDINGS_COLUMNS:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported embedding model: {model.value}",
            )
        models_to_process = [model]

    question_rows = []
    if all_questions:
        question_rows = await db.fetch("SELECT id, name, content FROM question")
    elif questions:
        placeholders = ",".join(["$" + str(i + 1) for i in range(len(questions))])
        question_rows = await db.fetch(
            f"SELECT id, name, content FROM question WHERE name IN ({placeholders})",  # noqa: S608
            *questions,
        )

    async def _gen() -> AsyncGenerator[str]:
        progress_counter = 0
        total_tasks = 0

        if question_rows:
            total_tasks = len(question_rows) * len(models_to_process)
        else:
            for m in models_to_process:
                col = MODEL_EMBEDDINGS_COLUMNS[m]
                count_result = await db.fetchval(
                    f"SELECT COUNT(*) FROM question WHERE {col} IS NULL",  # noqa: S608
                )
                if isinstance(count_result, int | str):
                    total_tasks += int(count_result)

        for current_model in models_to_process:
            model_column = MODEL_EMBEDDINGS_COLUMNS[current_model]

            rows_for_this_model = []
            if question_rows:
                rows_for_this_model = question_rows
            else:
                rows_for_this_model = await db.fetch(
                    f"SELECT id, name, content FROM question WHERE {model_column} IS NULL",  # noqa: S608
                )

            for row in rows_for_this_model:
                progress_counter += 1
                qid = row["id"]
                name = row["name"]
                result = "ok"
                error_detail = ""

                try:
                    document_text = f"Наслов: {name}\nСодржина: {row['content']}"
                    text_to_embed = f"преземи документ: {document_text}"

                    embedding = await generate_embeddings(text_to_embed, current_model)
                    await db.execute(
                        f"UPDATE question SET {model_column} = $1 WHERE id = $2",  # noqa: S608
                        embedding_to_pgvector(embedding),
                        qid,
                    )
                except Exception as e:
                    result = "error"
                    error_detail = repr(e)

                payload = {
                    "status": result,
                    "error": error_detail,
                    "index": progress_counter,
                    "total": total_tasks,
                    "model": current_model.value,
                    "id": str(qid),
                    "name": name,
                    "ts": datetime.now(UTC).isoformat() + "Z",
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
    )
