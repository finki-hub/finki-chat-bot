from typing import overload

import torch
from langchain_huggingface import HuggingFaceEmbeddings

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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _bge_m3_embedder = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    if isinstance(text, str):
        return _bge_m3_embedder.embed_query(text)
    return _bge_m3_embedder.embed_documents(text)
