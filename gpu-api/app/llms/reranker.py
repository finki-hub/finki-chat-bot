import torch
from sentence_transformers import CrossEncoder

_reranker_model: CrossEncoder | None = None


def init_reranker() -> None:
    """
    Initializes the BGE Re-ranker model during application startup.
    This function should be called from the lifespan manager.
    """
    global _reranker_model  # noqa: PLW0603
    if _reranker_model is None:
        model_name = "BAAI/bge-reranker-large"
        print(f"Initializing re-ranker model: {model_name}...")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        _reranker_model = CrossEncoder(model_name, device=device)

        print(f"Re-ranker model initialized successfully on device: '{device}'")


def rerank_documents(query: str, documents: list[str]) -> list[str]:
    """
    Re-ranks a list of documents based on their relevance to a query
    using the pre-loaded cross-encoder model.
    """
    if not documents or not query:
        return documents
    if _reranker_model is None:
        raise RuntimeError("Re-ranker model has not been initialized.")

    model_inputs = [[query, doc] for doc in documents]

    scores = _reranker_model.predict(model_inputs)

    scored_docs = sorted(
        zip(scores, documents, strict=False),
        key=lambda x: x[0],
        reverse=True,
    )

    return [doc for _, doc in scored_docs]
