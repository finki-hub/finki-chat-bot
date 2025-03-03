from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class LinkSchema(BaseModel):
    id: UUID
    name: str
    url: str
    description: str | None = Field(default=None)
    user_id: str | None = Field(default=None)
    created_at: datetime
    updated_at: datetime


class CreateLinkSchema(BaseModel):
    name: str
    url: str
    description: str | None = Field(default=None)
    user_id: str | None = Field(default=None)


class UpdateLinkSchema(BaseModel):
    name: str | None = Field(default=None)
    url: str | None = Field(default=None)
    description: str | None = Field(default=None)
    user_id: str | None = Field(default=None)
