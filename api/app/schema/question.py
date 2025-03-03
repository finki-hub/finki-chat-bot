from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class QuestionSchema(BaseModel):
    id: UUID
    name: str
    content: str
    user_id: str
    links: dict[str, str] | None
    created_at: datetime
    updated_at: datetime


class CreateQuestionSchema(BaseModel):
    name: str
    content: str
    user_id: str | None = Field(default=None)
    links: dict[str, str] | None = Field(default=None)


class UpdateQuestionSchema(BaseModel):
    name: str | None = Field(default=None)
    content: str | None = Field(default=None)
    user_id: str | None = Field(default=None)
    links: dict[str, str] | None = Field(default=None)
