from pydantic import BaseModel

from app.llms.models import Model


class ChatQuestion(BaseModel):
    question: str
    model: Model
