from enum import Enum


class Model(Enum):
    """
    Enum representing the available models.
    """

    BGE_M3 = "BAAI/bge-m3"
    MULTILINGUAL_E5_LARGE = "intfloat/multilingual-e5-large"
    QWEN2_1_5_B_INSTRUCT = "Qwen/Qwen2-1.5B-Instruct"
