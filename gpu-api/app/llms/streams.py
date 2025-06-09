import json
from collections.abc import AsyncGenerator

from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse

from app.llms.models import Model
from app.llms.qwen2_1_5_b_instruct import stream_qwen2_response

_streamer_map = {
    Model.QWEN2_1_5_B_INSTRUCT: stream_qwen2_response,
}


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
    Dispatches to the appropriate model streamer and wraps the output
    in a Server-Sent Events (SSE) response.
    """
    streamer = _streamer_map.get(model)
    if streamer is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model {model.value} is not supported for streaming chat.",
        )

    async def _sse_generator() -> AsyncGenerator[str]:
        async for token in streamer(
            user_prompt,
            system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        ):
            yield f"data: {token}\n\n"

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
    )
