from schema.models import Model

from .bgam3 import get_bgam3_embeddings

embeddings_map = {
    Model["BAAI/bge-m3"]: get_bgam3_embeddings,
}


def get_embeddings(text: str | list[str], model: Model):
    if model not in embeddings_map:
        raise ValueError(f"Model {model} is not supported for embeddings.")

    return embeddings_map[model](text)
