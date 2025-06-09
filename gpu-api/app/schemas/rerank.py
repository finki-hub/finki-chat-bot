from pydantic import BaseModel, Field


class RerankRequestSchema(BaseModel):
    query: str = Field(description="The original user query.")
    documents: list[str] = Field(
        description="A list of document contents to be re-ranked.",
    )


class RerankResponseSchema(BaseModel):
    reranked_documents: list[str] = Field(
        description="The documents re-ordered by their relevance to the query.",
    )
