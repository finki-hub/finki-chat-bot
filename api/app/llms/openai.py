import asyncio
from collections.abc import AsyncGenerator, Generator

from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.llms.models import Model
from app.llms.prompts import stitch_system_user
from app.utils.settings import Settings

settings = Settings()

_llm_clients_openai: dict[tuple[str, float, float, int], ChatOpenAI] = {}


def get_openai_llm(
    model: Model,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> ChatOpenAI:
    """
    Return a singleton ChatOpenAI instance for the specified model and sampling parameters.
    """
    key = (model.value, temperature, top_p, max_tokens)
    if key not in _llm_clients_openai:
        _llm_clients_openai[key] = ChatOpenAI(
            model=model.value,
            api_key=SecretStr(settings.OPENAI_API_KEY),
            temperature=temperature,
            top_p=top_p,
            streaming=True,
            max_tokens=max_tokens,  # type: ignore[call-arg]
        )
    return _llm_clients_openai[key]


async def stream_openai_response(
    user_prompt: str,
    model: Model,
    *,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> StreamingResponse:
    """
    Stream a response from the specified OpenAI model using the provided user prompt and system prompt.
    The response is formatted as Server-Sent Events (SSE) for real-time updates.
    """
    llm = get_openai_llm(model, temperature, top_p, max_tokens)
    full_prompt = stitch_system_user(system_prompt, user_prompt)

    def sync_token_gen() -> Generator[str]:
        for chunk in llm.stream(full_prompt):
            yield str(chunk.content)

    async def async_token_gen() -> AsyncGenerator[str]:
        it = sync_token_gen()
        try:
            while True:
                chunk = await asyncio.to_thread(next, it, None)
                if chunk is None:
                    break
                yield f"data: {chunk}\n\n"
        except asyncio.CancelledError:
            return

    return StreamingResponse(
        async_token_gen(),
        media_type="text/event-stream",
    )
