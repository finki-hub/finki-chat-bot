from fastapi.responses import StreamingResponse

from app.llms.models import Model
from app.llms.ollama import stream_ollama_response


async def generate_response(
    prompt: str,
    model: Model,
) -> StreamingResponse:
    match model:
        case Model.LLAMA_3_3_70B | Model.MISTRAL:
            return await stream_ollama_response(prompt, model)
        case _:
            raise ValueError(f"Unsupported model: {model}")
