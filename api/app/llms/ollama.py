import asyncio
from collections.abc import AsyncGenerator, Generator

from fastapi.responses import StreamingResponse
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from app.llms.models import Model
from app.llms.prompts import stitch_system_user
from app.utils.settings import Settings

settings = Settings()

_embed_clients: dict[Model, OllamaEmbeddings] = {}
_llm_clients: dict[tuple[str, float, float, int], OllamaLLM] = {}


def get_embedder(model: Model) -> OllamaEmbeddings:
    """
    Return a singleton OllamaEmbeddings instance for the specified model.
    If the model is not already in the cache, create a new instance.
    """
    if model not in _embed_clients:
        _embed_clients[model] = OllamaEmbeddings(
            model=model.value,
            base_url=settings.OLLAMA_URL,
        )
    return _embed_clients[model]


def get_llm(
    model: Model,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> OllamaLLM:
    """
    Return a singleton OllamaLLM instance for the specified model and sampling parameters.
    If the model and parameters are not already in the cache, create a new instance.
    """
    key = (model.value, temperature, top_p, max_tokens)
    if key not in _llm_clients:
        _llm_clients[key] = OllamaLLM(
            model=model.value,
            base_url=settings.OLLAMA_URL,
            temperature=temperature,
            top_p=top_p,
            num_predict=max_tokens,
        )
    return _llm_clients[key]


async def generate_ollama_embeddings(text: str, model: Model) -> list[float]:
    """
    Generate embeddings for the given text using the specified Ollama model.
    This function runs the embedding generation in a separate thread to avoid blocking the event loop.
    """
    emb = get_embedder(model)
    return await asyncio.to_thread(emb.embed_query, text)


async def stream_ollama_response(
    user_prompt: str,
    model: Model,
    *,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> StreamingResponse:
    """
    Stream a response from the specified Ollama model using the provided user prompt and system prompt.
    This function constructs the full prompt by stitching the system and user prompts together,
    initializes the LLM client, and streams the response as an async generator.
    The response is formatted as Server-Sent Events (SSE) for real-time updates.
    """
    llm = get_llm(model, temperature, top_p, max_tokens)

    full_prompt = stitch_system_user(system_prompt, user_prompt)

    def sync_token_gen() -> Generator[str]:
        yield from llm.stream(full_prompt)

    async def async_token_gen() -> AsyncGenerator[str]:
        it = sync_token_gen()
        try:
            while True:
                chunk = await asyncio.to_thread(next, it, None)
                if chunk is None:
                    break
                preserved_chunk = chunk.replace("\n", "\\n")
                yield f"data: {preserved_chunk}\n\n"
        except asyncio.CancelledError:
            return

    return StreamingResponse(
        async_token_gen(),
        media_type="text/event-stream",
    )
