import asyncio
from collections.abc import AsyncGenerator, Generator

from fastapi.responses import StreamingResponse
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from app.llms.models import Model
from app.utils.settings import Settings

settings = Settings()


async def generate_ollama_embeddings(text: str, model: Model) -> list[float]:
    ollama_embeddings = OllamaEmbeddings(
        model=model.value,
        base_url=settings.OLLAMA_URL,
    )

    return await asyncio.to_thread(ollama_embeddings.embed_query, text)


async def stream_ollama_response(prompt: str, model: Model) -> StreamingResponse:
    ollama_llm = OllamaLLM(model=model.value, base_url=settings.OLLAMA_URL)

    def sync_token_gen() -> Generator[str]:
        yield from ollama_llm.stream(prompt)

    async def async_token_gen() -> AsyncGenerator[str]:
        iterator = sync_token_gen()

        while True:
            chunk = await asyncio.to_thread(next, iterator, None)
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(async_token_gen(), media_type="text/plain")
