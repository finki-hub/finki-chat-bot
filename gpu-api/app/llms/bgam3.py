from langchain.embeddings import HuggingFaceEmbeddings

bgam3_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)


def get_bgam3_embeddings(text: str | list[str]):
    if isinstance(text, str):
        return bgam3_embeddings.embed_query(text)

    return bgam3_embeddings.embed_documents(text)
