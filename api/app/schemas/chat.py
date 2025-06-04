from pydantic import BaseModel, Field

from app.constants.defaults import (
    DEFAULT_EMBEDDINGS_MODEL,
    DEFAULT_INFERENCE_MODEL,
)
from app.llms.models import Model


class ChatRequestSchema(BaseModel):
    question: str = Field(
        examples=["Where is FINKI located?"],
        description="The user's free-text query to send to the chat system.",
    )
    embeddings_model: Model = Field(
        DEFAULT_EMBEDDINGS_MODEL,
        examples=[DEFAULT_EMBEDDINGS_MODEL.value],
        description=(
            "Which Model to use for computing embeddings for retrieval. "
            "Must be one of the values in `app.llms.models.Model`."
        ),
    )
    inference_model: Model = Field(
        DEFAULT_INFERENCE_MODEL,
        examples=[DEFAULT_INFERENCE_MODEL.value],
        description=(
            "Which Model to use for generating / streaming the response. "
            "Must be one of the values in `app.llms.models.Model`."
        ),
    )
