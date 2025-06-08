from enum import Enum


class Model(Enum):
    """
    Enum representing the available models for generating embeddings.
    """

    BGE_M3 = "bge-m3:latest"
    MULTILINGUAL_E5_LARGE = "multilingual-e5-large:latest"
