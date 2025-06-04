import asyncio
from collections.abc import AsyncGenerator, Generator

from fastapi.responses import StreamingResponse
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from app.llms.models import Model
from app.utils.settings import Settings

settings = Settings()

_embed_clients: dict[Model, OllamaEmbeddings] = {}
_llm_clients: dict[Model, OllamaLLM] = {}


def get_embedder(model: Model) -> OllamaEmbeddings:
    """
    Return a singleton OllamaEmbeddings for the given model,
    instantiating it on first use.
    """
    if model not in _embed_clients:
        _embed_clients[model] = OllamaEmbeddings(
            model=model.value,
            base_url=settings.OLLAMA_URL,
        )
    return _embed_clients[model]


def get_llm(model: Model) -> OllamaLLM:
    """
    Return a singleton OllamaLLM for the given model,
    instantiating it on first use.
    """
    if model not in _llm_clients:
        _llm_clients[model] = OllamaLLM(
            model=model.value,
            base_url=settings.OLLAMA_URL,
        )
    return _llm_clients[model]


async def generate_ollama_embeddings(text: str, model: Model) -> list[float]:
    """
    Offload the blocking embed_query call to a thread, but reuse
    the same OllamaEmbeddings instance each time.
    """
    emb = get_embedder(model)
    return await asyncio.to_thread(emb.embed_query, text)


async def stream_ollama_response(prompt: str, model: Model) -> StreamingResponse:
    """
    Stream from a singleton OllamaLLM instance, so we only pay the
    client-startup cost once per model.
    """
    llm = get_llm(model)

    def sync_token_gen() -> Generator[str]:
        yield from llm.stream(prompt)

    async def async_token_gen() -> AsyncGenerator[str]:
        iterator = sync_token_gen()
        while True:
            chunk = await asyncio.to_thread(next, iterator, None)
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(async_token_gen(), media_type="text/event-stream")
