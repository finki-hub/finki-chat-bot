import asyncio
from typing import overload

from fastapi import HTTPException, status

from app.llms.bge_m3 import get_bge_m3_embeddings
from app.llms.models import Model

_embeddings_map = {
    Model.BGE_M3: get_bge_m3_embeddings,
}


@overload
async def get_embeddings(
    texts: str,
    model: Model,
) -> list[float]: ...


@overload
async def get_embeddings(
    texts: list[str],
    model: Model,
) -> list[list[float]]: ...


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

    def _call() -> list[float] | list[list[float]]:
        return embedder(texts)

    return await asyncio.to_thread(_call)
