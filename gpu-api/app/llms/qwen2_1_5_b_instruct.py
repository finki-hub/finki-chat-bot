import logging
from collections.abc import AsyncGenerator

import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline

logger = logging.getLogger(__name__)

_pipelines: dict[tuple[float, float, int], HuggingFacePipeline] = {}


def get_qwen2_pipeline(
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> HuggingFacePipeline:
    """
    Retrieves or initializes a HuggingFace pipeline for the Qwen2-1.5B model
    with the specified parameters.
    """
    logger.info(
        "Initializing Qwen2-1.5B pipeline with temperature: %s, top_p: %s, max_tokens: %s",
        temperature,
        top_p,
        max_tokens,
    )

    model_id = "Qwen/Qwen2-1.5B-Instruct"
    key = (temperature, top_p, max_tokens)

    if key not in _pipelines:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

        _pipelines[key] = HuggingFacePipeline(pipeline=pipe)

    return _pipelines[key]


async def stream_qwen2_response(
    user_prompt: str,
    system_prompt: str,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> AsyncGenerator[str]:
    """
    Streams a response from the Qwen2-1.5B model using the specified parameters.
    """
    logger.info(
        "Streaming Qwen2-1.5B response for user prompt: %s",
        user_prompt,
    )

    llm = get_qwen2_pipeline(temperature, top_p, max_tokens)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt = llm.pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    async for chunk in llm.astream(prompt):
        yield chunk
