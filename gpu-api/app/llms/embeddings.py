import asyncio

from fastapi import HTTPException, status

from app.llms.bge_m3 import get_bge_m3_embeddings
from app.llms.models import Model

_embeddings_map = {
    Model.BGE_M3: get_bge_m3_embeddings,
}


async def get_embeddings(
    texts: str | list[str],
    model: Model,
) -> list[float] | list[list[float]]:
    """
    Dispatch to the appropriate embedder, offloading blocking calls.
    Raises HTTPException(400) if the model isn't supported.
    """
    embedder = _embeddings_map.get(model)
    if embedder is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model {model.value} is not supported for embeddings.",
        )
    return await asyncio.to_thread(embedder, texts)
