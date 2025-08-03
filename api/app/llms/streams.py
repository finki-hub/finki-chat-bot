import logging

from fastapi.responses import StreamingResponse

from app.llms.google import stream_google_agent_response, stream_google_response
from app.llms.gpu_api import stream_gpu_api_response
from app.llms.models import Model
from app.llms.ollama import stream_ollama_agent_response, stream_ollama_response
from app.llms.openai import stream_openai_agent_response, stream_openai_response

logger = logging.getLogger(__name__)


async def stream_response(
    user_prompt: str,
    model: Model,
    *,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> StreamingResponse:
    """
    Stream a response from the specified model using the provided user prompt and system prompt.
    """
    logger.info(
        "Streaming response for user prompt length: '%d' with model: %s",
        len(user_prompt),
        model.value,
    )

    match model:
        case (
            Model.LLAMA_3_3_70B
            | Model.MISTRAL
            | Model.DEEPSEEK_R1_70B
            | Model.QWEN2_5_72B
            | Model.DOMESTIC_YAK_8B_INSTRUCT_GGUF
            | Model.VEZILKALLM_GGUF
        ):
            return await stream_ollama_response(
                user_prompt,
                model,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

        case Model.GPT_4O_MINI | Model.GPT_4_1_MINI | Model.GPT_4_1_NANO:
            return await stream_openai_response(
                user_prompt,
                model,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

        case Model.GEMINI_2_5_FLASH_PREVIEW_05_20:
            return await stream_google_response(
                user_prompt,
                model,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

        case Model.QWEN2_1_5_B_INSTRUCT:
            return await stream_gpu_api_response(
                user_prompt,
                model,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

        case _:
            raise ValueError(f"Unsupported model: {model}")


async def stream_response_with_agent(
    user_prompt: str,
    model: Model,
    *,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> StreamingResponse:
    """
    Stream a response from the specified model using the provided user prompt and system prompt with agent.
    """
    logger.info(
        "Streaming response with agent for user prompt length: '%d' with model: %s",
        len(user_prompt),
        model.value,
    )

    match model:
        case (
            Model.LLAMA_3_3_70B
            | Model.MISTRAL
            | Model.DEEPSEEK_R1_70B
            | Model.QWEN2_5_72B
            | Model.DOMESTIC_YAK_8B_INSTRUCT_GGUF
            | Model.VEZILKALLM_GGUF
        ):
            return await stream_ollama_agent_response(
                user_prompt,
                model,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

        case Model.GPT_4O_MINI | Model.GPT_4_1_MINI | Model.GPT_4_1_NANO:
            return await stream_openai_agent_response(
                user_prompt,
                model,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

        case Model.GEMINI_2_5_FLASH_PREVIEW_05_20:
            return await stream_google_agent_response(
                user_prompt,
                model,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

        case _:
            raise ValueError(f"Unsupported model: {model}")
