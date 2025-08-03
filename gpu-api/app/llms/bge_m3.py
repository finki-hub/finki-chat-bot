import logging
from typing import overload

import torch
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

_bge_m3_embedder: HuggingFaceEmbeddings | None = None


@overload
def get_bge_m3_embeddings(text: str) -> list[float]: ...


@overload
def get_bge_m3_embeddings(text: list[str]) -> list[list[float]]: ...


def get_bge_m3_embeddings(text: str | list[str]) -> list[float] | list[list[float]]:
    """
    Get embeddings using the BGE M3 model from Hugging Face.
    """
    global _bge_m3_embedder  # noqa: PLW0603

    if _bge_m3_embedder is None:
        logger.info("Loading BGE M3 embeddings model...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _bge_m3_embedder = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    if isinstance(text, str):
        return _bge_m3_embedder.embed_query(text)

    return _bge_m3_embedder.embed_documents(text)


def init_bge_m3_embedder() -> None:
    """
    Initialize the BGE M3 embedder.
    This function is called during application startup to ensure the embedder is ready for use.
    """

    logger.info("Initializing BGE M3 embeddings model...")

    if _bge_m3_embedder is None:
        get_bge_m3_embeddings("Initialization check")
        logger.info("BGE M3 embeddings model initialized successfully.")
    else:
        logger.info("BGE M3 embeddings model is already initialized.")
