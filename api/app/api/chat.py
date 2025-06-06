from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from ollama import ResponseError

from app.data.connection import Database
from app.data.db import get_db
from app.data.questions import get_closest_questions
from app.llms.embeddings import generate_embeddings
from app.llms.models import Model
from app.llms.prompts import DEFAULT_SYSTEM_PROMPT, build_context, build_user_prompt
from app.llms.streams import stream_response
from app.schemas.chat import ChatRequestSchema

db_dep = Depends(get_db)

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
    dependencies=[db_dep],
)


@router.post(
    "/",
    summary="Stream a chat response",
    description=(
        "Compute an embedding for the incoming question, retrieve top-N "
        "similar questions for context, construct a prompt, and stream back "
        "the LLM's answer as a text stream."
    ),
    response_class=StreamingResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Chunked stream of SSE events",
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "example": "data: Hello\n\ndata: World\n\n",
                    },
                },
            },
        },
        status.HTTP_400_BAD_REQUEST: {
            "description": "Unsupported model or invalid request",
        },
        status.HTTP_504_GATEWAY_TIMEOUT: {
            "description": "LLM service unavailable",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "The LLM service is currently unavailable. Please try again later.",
                    },
                },
            },
        },
    },
    operation_id="chatWithModel",
)
async def chat(
    payload: ChatRequestSchema,
    db: Database = db_dep,
) -> StreamingResponse:
    try:
        prompt_embedding = await generate_embeddings(
            payload.prompt,
            payload.embeddings_model,
        )
        closest_questions = await get_closest_questions(
            db,
            prompt_embedding,
            payload.embeddings_model,
            limit=20,
        )
        context = build_context(closest_questions)
        user_prompt = build_user_prompt(context, payload.prompt)

        return await stream_response(
            user_prompt,
            payload.inference_model,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            temperature=payload.temperature,
            top_p=payload.top_p,
            max_tokens=payload.max_tokens,
        )
    except ResponseError as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The LLM service is currently unavailable. Please try again later.",
        ) from e


@router.get(
    "/models",
    summary="List available LLM models",
    description="Retrieve a list of all available LLM models for chat.",
    response_model=list[str],
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "List of available LLM models",
            "content": {
                "application/json": {
                    "example": ["llama3.3:70b", "qwen2.5:72b"],
                },
            },
        },
    },
)
def list_models() -> list[str]:
    return [m.value for m in Model]
