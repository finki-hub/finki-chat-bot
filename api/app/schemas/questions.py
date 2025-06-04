from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl

from app.constants.defaults import DEFAULT_EMBEDDINGS_MODEL
from app.llms.models import Model


class QuestionSchema(BaseModel):
    id: UUID = Field(
        examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
        description="Unique identifier for the question",
    )
    name: str = Field(
        examples=["reset-password"],
        description="Key or slug of the question",
    )
    content: str = Field(
        examples=["How do I reset my password if I forgot it?"],
        description="Full text of the question",
    )
    user_id: str | None = Field(
        default=None,
        examples=["198249751001563136"],
        description="Discord ID of the user who created this question",
    )
    links: dict[str, HttpUrl] | None = Field(
        default=None,
        examples=[{"google": "https://www.google.com"}],
        description="Optional map of link names to URLs associated",
    )
    created_at: datetime = Field(
        examples=["2025-06-05T14:48:00Z"],
        description="UTC timestamp when the question was created",
    )
    updated_at: datetime = Field(
        examples=["2025-06-06T09:24:00Z"],
        description="UTC timestamp when the question was last updated",
    )


class CreateQuestionSchema(BaseModel):
    name: str = Field(
        examples=["reset-password"],
        description="Unique key or slug for the question",
    )
    content: str = Field(
        examples=["How do I reset my password if I forgot it?"],
        description="Full text body of the question",
    )
    user_id: str | None = Field(
        default=None,
        examples=["198249751001563136"],
        description="Discord ID of the user creating this question",
    )
    links: dict[str, HttpUrl] | None = Field(
        default=None,
        examples=[{"faq": "https://example.com/faq"}],
        description="Optional links to associate with the question",
    )


class UpdateQuestionSchema(BaseModel):
    name: str | None = Field(
        default=None,
        examples=["reset-password-v2"],
        description="New key or slug for the question",
    )
    content: str | None = Field(
        default=None,
        examples=["How can I recover my forgotten password?"],
        description="Updated text of the question",
    )
    user_id: str | None = Field(
        default=None,
        examples=["198249751001563136"],
        description="Updated user Discord ID",
    )
    links: dict[str, HttpUrl] | None = Field(
        default=None,
        examples=[{"help": "https://example.com/help"}],
        description="Updated set of associated links",
    )


class EmbedQuestionsSchema(BaseModel):
    model: Model = Field(
        examples=[DEFAULT_EMBEDDINGS_MODEL.value],
        description="Which embedding model to use",
    )
    all: bool = Field(
        default=False,
        examples=[False],
        description="Whether to regenerate _all_ embeddings vs. only missing ones",
    )


class GetClosestQuestionsSchema(BaseModel):
    question: str = Field(
        examples=["What is the capital of France?"],
        description="Query string to embed and search",
    )
    model: Model = Field(
        examples=[DEFAULT_EMBEDDINGS_MODEL.value],
        description="Which embedding model to use for search",
    )
