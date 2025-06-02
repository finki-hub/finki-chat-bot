from fastapi import APIRouter
from pydantic import BaseModel
from schema.models import Model

from app.constants import strings
from app.llms.embeddings import get_embeddings

router = APIRouter(prefix="/embeddings", tags=["Embeddings"])


class EmbedRequestSchema(BaseModel):
    input: str | list[str]
    model: Model


class EmbedResponseSchema(BaseModel):
    embeddings: list[float] | list[list[float]]


@router.get("/check", response_model=str)
async def check() -> str:
    return strings.API_RUNNING


@router.get("/embed", response_model=EmbedResponseSchema)
async def embed(request: EmbedRequestSchema) -> EmbedResponseSchema:
    embeddings = get_embeddings(request.input, request.model)
    return EmbedResponseSchema(embeddings=embeddings)
