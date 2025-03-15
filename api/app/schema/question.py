from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class QuestionSchema(BaseModel):
    id: UUID
    name: str
    content: str
    user_id: str | None = None
    links: dict[str, str] | None = None
    created_at: datetime
    updated_at: datetime


class CreateQuestionSchema(BaseModel):
    name: str
    content: str
    user_id: str | None = None
    links: dict[str, str] | None = None


class UpdateQuestionSchema(BaseModel):
    name: str | None = None
    content: str | None = None
    user_id: str | None = None
    links: dict[str, str] | None = None
