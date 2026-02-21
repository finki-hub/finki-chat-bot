import asyncio
import logging
from collections.abc import AsyncGenerator, Generator
from typing import overload

from fastapi.responses import StreamingResponse
from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph.state import CompiledStateGraph
from pydantic import SecretStr

from app.llms.mcp import build_mcp_client
from app.llms.models import Model
from app.utils.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

# Model, temperature, top_p, max_tokens -> LLM
google_llm_clients: dict[tuple[str, float, float, int], ChatGoogleGenerativeAI] = {}
google_embedders: dict[str, GoogleGenerativeAIEmbeddings] = {}


def get_google_embedder(model: Model) -> GoogleGenerativeAIEmbeddings:
    """
    Return a singleton GoogleGenerativeAIEmbeddings instance for the specified model.
    If the model is not already in the cache, create a new instance.
    """
    key = model.value

    if key not in google_embedders:
        google_embedders[key] = GoogleGenerativeAIEmbeddings(
            model=model.value,
            api_key=SecretStr(settings.GOOGLE_API_KEY),
        )

    return google_embedders[key]


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

    if key not in google_llm_clients:
        google_llm_clients[key] = ChatGoogleGenerativeAI(
            model=model.value,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )

    return google_llm_clients[key]


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
    logger.info(
        "Generating Google embeddings for text with length '%s' with model: %s",
        len(text) if isinstance(text, str) else sum(len(t) for t in text),
        model.value,
    )

    emb = get_google_embedder(model)

    if isinstance(text, str):
        return await asyncio.to_thread(emb.embed_query, text)

    return await asyncio.to_thread(emb.embed_documents, text)


def stream_google_response(
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
    logger.info(
        "Streaming Google response for user prompt length: '%d' with model: %s",
        len(user_prompt),
        model.value,
    )

    llm = get_google_llm(model, temperature, top_p, max_tokens)
    prompt_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    def sync_token_gen() -> Generator[str]:
        for chunk in llm.stream(prompt_messages):
            content = chunk.content
            if isinstance(content, list):
                text = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            else:
                text = str(content)
            yield text

    async def async_token_gen() -> AsyncGenerator[str]:
        it = sync_token_gen()
        while True:
            chunk = await asyncio.to_thread(next, it, None)
            if chunk is None:
                break
            preserved_chunk = chunk.replace("\n", "\\n")
            yield f"data: {preserved_chunk}\n\n"

    return StreamingResponse(
        async_token_gen(),
        media_type="text/event-stream",
    )


async def _create_agent_token_generator(
    agent: CompiledStateGraph,
    messages: list[dict[str, str]],
) -> AsyncGenerator[str]:
    """Helper function to generate tokens from agent stream."""
    try:
        async for message, _metadata in agent.astream(
            {"messages": messages},
            {"configurable": {"thread_id": "default"}},
            stream_mode="messages",
        ):
            if not isinstance(message, AIMessageChunk):
                continue
            raw = message.content
            if isinstance(raw, list):
                text = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in raw
                )
            else:
                text = str(raw)
            if not text:
                continue
            preserved = text.replace("\n", "\\n")
            yield f"data: {preserved}\n\n"

    except Exception:
        logger.exception("Agent error occurred during streaming")
        yield "data: An error occurred while processing your request. Please try again.\n\n"


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
    logger.info(
        "Streaming Google agent response for user prompt length: '%d' with model: %s",
        len(user_prompt),
        model.value,
    )

    try:
        llm = get_google_llm(model, temperature, top_p, max_tokens)

        client = build_mcp_client()
        tools = await client.get_tools()

        logger.info(
            "Available tools: %s",
            ", ".join(tool.name for tool in tools) if tools else "None",
        )

        agent = create_agent(llm, tools)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return StreamingResponse(
            _create_agent_token_generator(agent, messages),
            media_type="text/event-stream",
        )

    except Exception:
        logger.exception(
            "Failed to stream Google agent response. Falling back to regular response",
        )

        return stream_google_response(
            user_prompt,
            model,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
