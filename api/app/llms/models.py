from enum import Enum


class Model(Enum):
    """
    Enum representing the available models for inference and embeddings.
    """

    LLAMA_3_3_70B = "llama3.3:70b"
    MISTRAL = "mistral:latest"
    DEEPSEEK_R1_70B = "deepseek-r1:70b"
    QWEN2_5_72B = "qwen2.5:72b"
    BGE_M3 = "bge-m3:latest"

    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"

    GEMINI_2_5_FLASH_PREVIEW_05_20 = "gemini-2.5-flash-preview-05-20"
    TEXT_EMBEDDING_004 = "models/text-embedding-004"

    DOMESTIC_YAK_8B_INSTRUCT_GGUF = "hf.co/LVSTCK/domestic-yak-8B-instruct-GGUF:Q8_0"
    VEZILKALLM_GGUF = "hf.co/mradermacher/VezilkaLLM-GGUF:Q8_0"

    MULTILINGUAL_E5_LARGE = "intfloat/multilingual-e5-large"
    QWEN2_1_5_B_INSTRUCT = "Qwen/Qwen2-1.5B-Instruct"


MODEL_EMBEDDINGS_COLUMNS: dict[Model, str] = {
    Model.LLAMA_3_3_70B: "embedding_llama3_3_70b",
    Model.BGE_M3: "embedding_bge_m3",
    Model.TEXT_EMBEDDING_3_LARGE: "embedding_text_embedding_3_large",
    Model.TEXT_EMBEDDING_004: "embedding_text_embedding_004",
    Model.MULTILINGUAL_E5_LARGE: "embedding_multilingual_e5_large",
}
