import logging

from fastapi.responses import StreamingResponse

from app.llms.prompts import (
    DEFAULT_AGENT_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    build_user_agent_prompt,
    build_user_prompt,
)
from app.llms.streams import stream_response, stream_response_with_agent
from app.schemas.chat import ChatSchema

logger = logging.getLogger(__name__)


async def handle_regular_chat(
    payload: ChatSchema,
    context: str,
) -> StreamingResponse:
    """
    Handle regular chat without MCP tools.
    """
    logger.info(
        "Handling regular chat for user prompt: '%s' with model: %s",
        payload.prompt,
        payload.inference_model.value,
    )

    system_prompt = payload.system_prompt or DEFAULT_SYSTEM_PROMPT
    user_prompt = build_user_prompt(context, payload.prompt)

    return await stream_response(
        user_prompt,
        payload.inference_model,
        system_prompt=system_prompt,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
    )


async def handle_agent_chat(
    payload: ChatSchema,
    context: str,
) -> StreamingResponse:
    """
    Handle chat with MCP tools using an agent.
    """
    logger.info(
        "Handling agent chat for user prompt: '%s' with model: %s",
        payload.prompt,
        payload.inference_model.value,
    )

    system_prompt = payload.system_prompt or DEFAULT_AGENT_SYSTEM_PROMPT
    user_prompt = build_user_agent_prompt(context, payload.prompt)

    return await stream_response_with_agent(
        user_prompt,
        payload.inference_model,
        system_prompt=system_prompt,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
    )
