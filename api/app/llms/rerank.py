import httpx

from app.data.connection import Database
from app.data.questions import get_closest_questions
from app.llms.embeddings import generate_embeddings
from app.llms.models import Model
from app.utils.settings import Settings

settings = Settings()


class RetrievalError(Exception):
    """Custom exception for retrieval or re-ranking failures."""


async def get_retrieved_context(
    db: Database,
    query: str,
    embedding_model: Model,
    *,
    use_reranker: bool,
    initial_k: int = 20,
    top_k: int = 5,
) -> str:
    """
    Performs a retrieval process. If use_reranker is True, it's a two-stage
    process (vector search + re-ranking). Otherwise, it's a single-stage
    vector search.
    """
    print(
        f"Starting retrieval for query: '{query[:50]}...'. Reranker enabled: {use_reranker}",
    )

    retrieval_limit = initial_k if use_reranker else top_k

    try:
        query_to_embed = f"пребарување: {query}"
        prompt_embedding = await generate_embeddings(query_to_embed, embedding_model)
        initial_candidates = await get_closest_questions(
            db,
            prompt_embedding,
            embedding_model,
            limit=retrieval_limit,
        )
        if not initial_candidates:
            return ""
    except Exception as e:
        raise RetrievalError(f"Failed during initial vector search: {e}") from e

    candidate_docs = [
        f"Наслов: {q.name}\nСодржина: {q.content}" for q in initial_candidates
    ]

    if use_reranker:
        try:
            print(f"Sending {len(candidate_docs)} candidates to re-ranker...")
            rerank_payload = {"query": query, "documents": candidate_docs}
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{settings.GPU_API_URL}/rerank/",
                    json=rerank_payload,
                )
                response.raise_for_status()
                final_docs = response.json()["reranked_documents"]
            print(
                f"Re-ranking completed. Selected top {len(final_docs)} documents.",
            )
        except Exception as e:
            print(
                f"Re-ranking call failed: {e}. Using vector search order as a fallback.",
            )
            final_docs = candidate_docs
    else:
        final_docs = candidate_docs

    return "\n\n---\n\n".join(final_docs[:top_k])
