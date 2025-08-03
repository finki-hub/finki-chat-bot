import asyncio
import logging

from fastapi import APIRouter, HTTPException, status

from app.llms.reranker import rerank_documents
from app.schemas.rerank import RerankRequestSchema, RerankResponseSchema
from app.utils.exceptions import ModelNotReadyError

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/rerank",
    tags=["Re-ranking"],
)


@router.post(
    "/",
    summary="Re-rank documents based on a query",
    description=(
        "Accepts a query and a list of documents, and returns them re-ordered "
        "by their semantic relevance to the query."
    ),
    response_model=RerankResponseSchema,
    status_code=status.HTTP_200_OK,
    operation_id="rerankDocuments",
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "An unexpected error occurred during the re-ranking process.",
            "content": {
                "application/json": {
                    "example": {"detail": "An unexpected error occurred."},
                },
            },
        },
    },
)
async def handle_rerank(payload: RerankRequestSchema) -> RerankResponseSchema:
    logger.info(
        "Received rerank request with query: %s and %d documents",
        payload.query,
        len(payload.documents),
    )

    if not payload.documents:
        return RerankResponseSchema(reranked_documents=[])

    try:
        reranked_list = await asyncio.to_thread(
            rerank_documents,
            payload.query,
            payload.documents,
        )

        return RerankResponseSchema(reranked_documents=reranked_list)
    except ModelNotReadyError as e:
        logger.exception("Reranker model has not been initialized.")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reranker model has not been initialized.",
        ) from e
    except Exception as e:
        logger.exception("Error during reranking documents")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during the reranking process.",
        ) from e
