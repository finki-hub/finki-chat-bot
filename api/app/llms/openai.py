import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import overload

from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent
from pydantic import SecretStr

from app.llms.mcp import get_mcp_client
from app.llms.models import Model
from app.llms.prompts import stitch_system_user
from app.utils.settings import Settings

settings = Settings()

_llm_clients_openai: dict[tuple[str, float, float, int], ChatOpenAI] = {}
_openai_embedders: dict[str, OpenAIEmbeddings] = {}


def get_openai_embedder(model: Model) -> OpenAIEmbeddings:
    """
    Return a singleton OpenAIEmbeddings instance for the specified model.
    If the model is not already in the cache, create a new instance.
    """
    key = model.value
    if key not in _openai_embedders:
        _openai_embedders[key] = OpenAIEmbeddings(
            model=model.value,
            api_key=SecretStr(settings.OPENAI_API_KEY),  # type: ignore[call-arg]
        )
    return _openai_embedders[key]


def get_openai_llm(
    model: Model,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> ChatOpenAI:
    """
    Return a singleton ChatOpenAI instance for the specified model and sampling parameters.
    If the model and parameters are not already in the cache, create a new instance.
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


@overload
async def generate_openai_embeddings(
    text: str,
    model: Model,
) -> list[float]: ...


@overload
async def generate_openai_embeddings(
    text: list[str],
    model: Model,
) -> list[list[float]]: ...


async def generate_openai_embeddings(
    text: str | list[str],
    model: Model,
) -> list[float] | list[list[float]]:
    """
    Generate embeddings for the given text using the specified OpenAI model.
    This function runs the embedding generation in a separate thread to avoid blocking the event loop.
    """
    emb = get_openai_embedder(model)
    if isinstance(text, str):
        return await asyncio.to_thread(emb.embed_query, text)
    return await asyncio.to_thread(emb.embed_documents, text)


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
                preserved_chunk = chunk.replace("\n", "\\n")
                yield f"data: {preserved_chunk}\n\n"
        except asyncio.CancelledError:
            return

    return StreamingResponse(
        async_token_gen(),
        media_type="text/event-stream",
    )


async def stream_openai_agent_response(
    user_prompt: str,
    model: Model,
    *,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> StreamingResponse:
    """
    Stream a response from an OpenAI agent with MCP tools.
    Falls back to regular response if MCP unavailable.
    """
    try:
        llm = get_openai_llm(model, temperature, top_p, max_tokens)

        client = await get_mcp_client()
        print(client.__dict__)
        tools = await client.get_tools()

        if not tools:
            return await stream_openai_response(
                user_prompt,
                model,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

        agent = create_react_agent(llm, tools)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        async def async_token_gen() -> AsyncGenerator[str]:
            try:
                async for chunk in agent.astream(
                    {"messages": messages},
                    {"configurable": {"thread_id": "default"}},
                ):
                    if "agent" in chunk:
                        agent_messages = chunk["agent"]["messages"]
                        for message in agent_messages:
                            if hasattr(message, "content") and message.content:
                                content = str(message.content)
                                preserved_content = content.replace("\n", "\\n")
                                yield f"data: {preserved_content}\n\n"

            except Exception as e:
                error_msg = f"Agent error: {e!s}"
                yield f"data: {error_msg}\n\n"

        return StreamingResponse(
            async_token_gen(),
            media_type="text/event-stream",
        )

    except Exception:
        return await stream_openai_response(
            user_prompt,
            model,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
