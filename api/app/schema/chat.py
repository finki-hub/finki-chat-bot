from pydantic import BaseModel, Field

from app.llms.models import Model


class ChatQuestion(BaseModel):
    question: str
    embeddings_model: Model = Field(default=Model.BGE_M3)
    inference_model: Model = Field(default=Model.MISTRAL)
