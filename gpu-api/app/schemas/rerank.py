from pydantic import BaseModel, Field


class RerankRequestSchema(BaseModel):
    query: str = Field(description="The original user query.")
    documents: list[str] = Field(
        description="A list of document contents to be reranked.",
    )


class RerankResponseSchema(BaseModel):
    reranked_documents: list[str] = Field(
        description="The documents reordered by their relevance to the query.",
    )
