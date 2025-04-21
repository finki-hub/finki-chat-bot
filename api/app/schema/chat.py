from pydantic import BaseModel

from app.utils.models import Model


class ChatQuestion(BaseModel):
    question: str
    model: Model
