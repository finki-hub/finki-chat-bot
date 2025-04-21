from enum import Enum


class Model(Enum):
    """
    Enum class to represent different model types.
    """

    LLAMA_3_3_70B = "llama3.3:70b"


MODEL_COLUMNS: dict[Model, str] = {
    Model.LLAMA_3_3_70B: "embedding_llama3_3_70b",
}
