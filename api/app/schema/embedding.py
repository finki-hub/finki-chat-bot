from pydantic import BaseModel, Field

from app.utils.models import Model


class EmbeddingOptions(BaseModel):
    model: Model
    all: bool = Field(default=False)
