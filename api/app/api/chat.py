import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from httpx import HTTPStatusError
from ollama import ResponseError

from app.data.connection import Database
from app.data.db import get_db
from app.llms.chat import handle_agent_chat, handle_regular_chat
from app.llms.context import RetrievalError, get_retrieved_context
from app.llms.models import Model
from app.schemas.chat import ChatSchema
from app.utils.exceptions import ModelNotReadyError

logger = logging.getLogger(__name__)

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
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Model not ready or retrieval error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "The model is not ready. Please try again later.",
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
    logger.info(
        "Received chat request with payload: %s",
        payload.model_dump(mode="json", exclude_defaults=True),
    )

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
        return handle_regular_chat(payload, context)

    except ModelNotReadyError as e:
        logger.exception("Model not ready for chat request: %s")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The model is not ready. Please try again later.",
        ) from e
    except RetrievalError as e:
        logger.exception(
            "Error retrieving or re-ranking context for query '%s'",
            payload.prompt,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to retrieve or re-rank context for the query.",
        ) from e
    except HTTPStatusError as e:
        logger.exception(
            "HTTP error occurred while processing the chat request: %s",
            e.response.text,
        )

        raise HTTPException(
            status_code=e.response.status_code,
            detail=e.response.json().get(
                "detail",
                "An error occurred while processing the request.",
            ),
        ) from e
    except ResponseError as e:
        logger.exception(
            "Error communicating with the LLM service",
        )

        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The LLM service is currently unavailable. Please try again later.",
        ) from e
    except Exception as e:
        logger.exception(
            "An unexpected error occurred while processing the chat request",
        )

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
