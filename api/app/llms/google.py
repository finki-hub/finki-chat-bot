import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import overload

from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.prebuilt import create_react_agent
from pydantic import SecretStr

from app.llms.mcp import build_mcp_client
from app.llms.models import Model
from app.utils.settings import Settings

settings = Settings()

_llm_clients_google: dict[tuple[str, float, float, int], ChatGoogleGenerativeAI] = {}

_google_embedders: dict[str, GoogleGenerativeAIEmbeddings] = {}


def get_google_embedder(model: Model) -> GoogleGenerativeAIEmbeddings:
    """
    Return a singleton GoogleGenerativeAIEmbeddings instance for the specified model.
    """
    key = model.value
    if key not in _google_embedders:
        _google_embedders[key] = GoogleGenerativeAIEmbeddings(
            model=model.value,
            google_api_key=SecretStr(settings.GOOGLE_API_KEY),
        )
    return _google_embedders[key]


def get_google_llm(
    model: Model,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> ChatGoogleGenerativeAI:
    """
    Return a singleton ChatGoogleGenerativeAI instance for the specified model.
    If the model and parameters are not already in the cache, create a new instance.
    """
    key = (model.value, temperature, top_p, max_tokens)
    if key not in _llm_clients_google:
        _llm_clients_google[key] = ChatGoogleGenerativeAI(
            model=model.value,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )
    return _llm_clients_google[key]


@overload
async def generate_google_embeddings(
    text: str,
    model: Model,
) -> list[float]: ...


@overload
async def generate_google_embeddings(
    text: list[str],
    model: Model,
) -> list[list[float]]: ...


async def generate_google_embeddings(
    text: str | list[str],
    model: Model,
) -> list[float] | list[list[float]]:
    """
    Generate embeddings for the given text using the specified Google model.
    This function runs the embedding generation in a separate thread to avoid
    blocking the event loop, mirroring the OpenAI implementation.
    """
    emb = get_google_embedder(model)
    if isinstance(text, str):
        return await asyncio.to_thread(emb.embed_query, text)
    return await asyncio.to_thread(emb.embed_documents, text)


async def stream_google_response(
    user_prompt: str,
    model: Model,
    *,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> StreamingResponse:
    """
    Stream a response from the specified Google model using the provided prompts.
    The response is formatted as Server-Sent Events (SSE).
    This function is a direct parallel to stream_openai_response.
    """
    llm = get_google_llm(model, temperature, top_p, max_tokens)
    prompt_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    def sync_token_gen() -> Generator[str]:
        for chunk in llm.stream(prompt_messages):
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


async def stream_google_agent_response(
    user_prompt: str,
    model: Model,
    *,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> StreamingResponse:
    """
    Stream a response from a Google agent with MCP tools.
    Falls back to regular response if MCP unavailable.
    """
    try:
        llm = get_google_llm(model, temperature, top_p, max_tokens)

        client = build_mcp_client()
        tools = await client.get_tools()

        if not tools:
            return await stream_google_response(
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
        return await stream_google_response(
            user_prompt,
            model,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
