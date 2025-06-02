from enum import Enum


class Model(Enum):
    """
    Enum class to represent different model types.
    """

    LLAMA_3_3_70B = "llama3.3:70b"
    BGE_M3 = "bge-m3:latest"


MODEL_COLUMNS: dict[Model, str] = {
    Model.LLAMA_3_3_70B: "embedding_llama3_3_70b",
    Model.BGE_M3: "embedding_bge_m3",
}
