from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from ollama import ResponseError

from app.data.connection import Database
from app.data.db import get_db
from app.llms.chat import handle_agent_chat, handle_regular_chat
from app.llms.models import Model
from app.llms.rerank import RetrievalError, get_retrieved_context
from app.schemas.chat import ChatSchema

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
    payload: ChatSchema,
    db: Database = db_dep,
) -> StreamingResponse:
    try:
        context = await get_retrieved_context(
            db=db,
            query=payload.prompt,
            embedding_model=payload.embeddings_model,
            use_reranker=payload.rerank_documents,
        )

        if not context:
            context = (
                "Не можев да пронајдам релевантни информации во базата на податоци."
            )

        if payload.use_agent:
            return await handle_agent_chat(payload, context)
        return await handle_regular_chat(payload, context)

    except RetrievalError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to retrieve or re-rank context for the query.",
        ) from e
    except ResponseError as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The LLM service is currently unavailable. Please try again later.",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal server error occurred.",
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
