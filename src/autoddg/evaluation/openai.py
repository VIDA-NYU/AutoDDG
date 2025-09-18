from __future__ import annotations

import openai
from beartype import beartype

from .base import BaseEvaluator


@beartype
class GPTEvaluator(BaseEvaluator):
    """
    Evaluate descriptions using OpenAI-compatible GPT models
    """

    def __init__(self, gpt4_api_key: str = "", model_name: str = "gpt-4o") -> None:
        client = openai.OpenAI(api_key=gpt4_api_key)
        super().__init__(client=client, model_name=model_name)


@beartype
class LLaMAEvaluator(BaseEvaluator):
    """
    Evaluate descriptions using DeepInfra LLaMA
    """

    def __init__(
        self,
        llama_api_key: str = "",
        model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct",
    ) -> None:
        client = openai.OpenAI(
            api_key=llama_api_key, base_url="https://api.deepinfra.com/v1/openai"
        )
        super().__init__(client=client, model_name=model_name)
