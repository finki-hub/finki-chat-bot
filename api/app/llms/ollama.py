from langchain_ollama import OllamaEmbeddings, OllamaLLM

from app.utils.config import OLLAMA_URL
from app.utils.models import Model


def generate_ollama_embeddings(text: str, model: Model) -> list[float]:
    ollama_embeddings = OllamaEmbeddings(
        model=model.value,
        base_url=OLLAMA_URL,
    )

    return ollama_embeddings.embed_query(text)


def generate_ollama_response(text: str, model: Model) -> str:
    ollama_llm = OllamaLLM(
        model=model.value,
        base_url=OLLAMA_URL,
    )

    return ollama_llm.invoke(text)
