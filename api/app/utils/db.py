def embedding_to_pgvector(embedding: list[float]) -> str:
    return "[" + ",".join(str(x) for x in embedding) + "]"
