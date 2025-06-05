import torch
from langchain_huggingface import HuggingFaceEmbeddings

_bge: HuggingFaceEmbeddings | None = None


def get_bge_m3_embeddings(text: str | list[str]) -> list[float] | list[list[float]]:
    """
    Get embeddings using the BGE M3 model from Hugging Face.
    """
    global _bge  # noqa: PLW0603
    if _bge is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _bge = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    if isinstance(text, str):
        return _bge.embed_query(text)
    return _bge.embed_documents(text)
