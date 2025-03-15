from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class LinkSchema(BaseModel):
    id: UUID
    name: str
    url: str
    description: str | None = None
    user_id: str | None = None
    created_at: datetime
    updated_at: datetime


class CreateLinkSchema(BaseModel):
    name: str
    url: str
    description: str | None = None
    user_id: str | None = None


class UpdateLinkSchema(BaseModel):
    name: str | None = None
    url: str | None = None
    description: str | None = None
    user_id: str | None = None
