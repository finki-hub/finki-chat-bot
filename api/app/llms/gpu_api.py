import asyncio
from collections.abc import AsyncGenerator

import httpx
from fastapi.responses import StreamingResponse

from app.llms.models import GPU_API_MODELS, Model
from app.utils.settings import Settings

settings = Settings()


class GpuApiError(Exception):
    """
    Custom exception for errors related to the GPU API service.
    """


async def generate_gpu_api_embeddings(
    text: str | list[str],
    model: Model,
) -> list[float] | list[list[float]]:
    """
    Generate embeddings using the GPU API service.
    """
    gpu_api_url = f"{settings.GPU_API_URL}/embeddings/embed"

    payload = {
        "input": text,
        "embeddings_model": GPU_API_MODELS[model],
    }

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                gpu_api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            response.raise_for_status()

            result = response.json()
            embeddings = result.get("embeddings")

            return embeddings

    except httpx.HTTPStatusError as e:
        raise GpuApiError(
            f"GPU API returned an error: {e.response.status_code} - {e.response.text}",
        ) from e
    except httpx.RequestError as e:
        raise GpuApiError(f"Connection error to GPU API: {e}") from e
    except Exception as e:
        raise GpuApiError(f"An unexpected error occurred calling GPU API: {e}") from e


async def stream_gpu_api_response(
    user_prompt: str,
    model: Model,
    *,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> StreamingResponse:
    """
    Stream a response from the GPU API service.
    """
    gpu_api_url = f"{settings.GPU_API_URL}/stream/"

    payload = {
        "prompt": user_prompt,
        "inference_model": model.value,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    async def stream_from_gpu_api() -> AsyncGenerator[str]:
        try:
            async with (
                httpx.AsyncClient(timeout=300) as client,
                client.stream(
                    "POST",
                    gpu_api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response,
            ):
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield f"data: Error from GPU API: {error_text.decode()}\n\n"
                    return

                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk.decode("utf-8")

        except httpx.RequestError as e:
            yield f"data: Connection error to GPU API: {e!s}\n\n"
        except asyncio.CancelledError:
            return
        except Exception as e:
            yield f"data: Unexpected error: {e!s}\n\n"

    return StreamingResponse(
        stream_from_gpu_api(),
        media_type="text/event-stream",
    )
