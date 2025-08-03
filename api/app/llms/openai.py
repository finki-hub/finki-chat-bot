import asyncio
import logging
from collections.abc import AsyncGenerator, Generator
from typing import overload

from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent
from pydantic import SecretStr

from app.llms.mcp import build_mcp_client
from app.llms.models import Model
from app.llms.prompts import stitch_system_user
from app.utils.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

# Model, temperature, top_p, max_tokens -> LLM
openai_llm_clients: dict[tuple[str, float, float, int], ChatOpenAI] = {}
openai_embedders: dict[str, OpenAIEmbeddings] = {}


def get_openai_embedder(model: Model) -> OpenAIEmbeddings:
    """
    Return a singleton OpenAIEmbeddings instance for the specified model.
    If the model is not already in the cache, create a new instance.
    """
    key = model.value

    if key not in openai_embedders:
        openai_embedders[key] = OpenAIEmbeddings(
            model=model.value,
            api_key=SecretStr(settings.OPENAI_API_KEY),  # type: ignore[call-arg]
        )

    return openai_embedders[key]


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

    if key not in openai_llm_clients:
        openai_llm_clients[key] = ChatOpenAI(
            model=model.value,
            api_key=SecretStr(settings.OPENAI_API_KEY),
            temperature=temperature,
            top_p=top_p,
            streaming=True,
            max_tokens=max_tokens,  # type: ignore[call-arg]
        )

    return openai_llm_clients[key]


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
    logger.info(
        "Generating OpenAI embeddings for text with length '%s' with model: %s",
        len(text) if isinstance(text, str) else sum(len(t) for t in text),
        model.value,
    )

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
    logger.info(
        "Streaming OpenAI response for user prompt: '%s' with model: %s",
        user_prompt,
        model.value,
    )

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
    logger.info(
        "Streaming OpenAI agent response for user prompt: '%s' with model: %s",
        user_prompt,
        model.value,
    )

    try:
        llm = get_openai_llm(model, temperature, top_p, max_tokens)

        client = build_mcp_client()
        tools = await client.get_tools()

        logger.info(
            "Available tools: %s",
            ", ".join(tool.name for tool in tools) if tools else "None",
        )

        if not tools:
            logger.warning(
                "No tools available for the agent. Falling back to regular response",
            )

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
        logger.exception(
            "Failed to stream OpenAI agent response. Falling back to regular response",
        )

        return await stream_openai_response(
            user_prompt,
            model,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )


async def transform_query_with_openai(
    query: str,
    model: Model,
    *,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    """
    Transform a query using the specified OpenAI model and system prompt.
    If the transformation fails, return the original query.
    """

    logger.info(
        "Transforming query: '%s' with model: %s",
        query,
        model,
    )

    try:
        llm = get_openai_llm(model, temperature, top_p, max_tokens)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        response = await llm.ainvoke(messages)
        return response.content.strip()  # type: ignore[union-attr]
    except Exception:
        logger.exception("Query transformation failed: %s. Using original query.")

        return query
