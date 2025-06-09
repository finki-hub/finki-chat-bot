from pydantic import BaseModel, Field

from app.llms.models import Model


class EmbedRequestSchema(BaseModel):
    input: str | list[str] = Field(
        description="A single string or list of strings to embed",
        min_length=1,
        examples=[
            "What is the capital of France?",
            ["Hello world", "FastAPI is great"],
        ],
    )
    embeddings_model: Model = Field(
        description="Which embedding model to use",
        examples=[Model.BGE_M3.value],
    )


class EmbedResponseSchema(BaseModel):
    embeddings: list[float] | list[list[float]] = Field(
        description="A single embedding vector or list of vectors",
        examples=[[0.12, -0.05, 0.34, 0.99], [[0.11, 0.22], [0.33, 0.44]]],
    )
