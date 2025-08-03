import logging

from fastapi import APIRouter, HTTPException, status

from app.llms.bge_m3 import ModelNotReadyError
from app.llms.embeddings import generate_embeddings
from app.schemas.embeddings import EmbedRequestSchema, EmbedResponseSchema

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/embeddings",
    tags=["Embeddings"],
)


@router.post(
    "/embed",
    summary="Generate embeddings",
    description="Given text(s) and a model, return the embedding vector(s).",
    response_model=EmbedResponseSchema,
    status_code=status.HTTP_200_OK,
    response_description="The embedding(s) as a list of floats",
    operation_id="embedText",
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "Unsupported model or invalid input",
        },
    },
)
async def embed(payload: EmbedRequestSchema) -> EmbedResponseSchema:
    try:
        embeddings = await generate_embeddings(payload.input, payload.embeddings_model)
        return EmbedResponseSchema(embeddings=embeddings)
    except ModelNotReadyError as e:
        logger.exception("Model is not ready for embeddings generation.")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.exception("Error generating embeddings: %s")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating embeddings",
        ) from e
