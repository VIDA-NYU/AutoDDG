from __future__ import annotations

from typing import Any

from beartype import beartype

from ..utils import load_prompts


@beartype
class BaseEvaluator:
    """Base class that implements the shared evaluation workflow

    Needs to be inherited through any custom evaluator implementations

    Args:
        client: LLM client instance
        model_name: Model name to use for evaluation
    """

    def __init__(self, client: Any, model_name: str) -> None:
        self.client = client
        self.model = model_name
        prompts = load_prompts()["evaluation"]
        self._system_message = prompts["system_message"].strip()
        self._evaluation_prompt = prompts["user_prompt"]

    def _build_content(self, description: str) -> str:
        return (
            f"{self._evaluation_prompt}\n"
            f"Description: {description}\n"
            "Evaluation Form (scores ONLY): "
        )

    def evaluate(self, description: str) -> str:
        """
        Evaluate the given description text & Return the raw scoring response from the model

        Args:
            description: Description text

        Returns:
            Evaluation response
        """

        content = self._build_content(description)
        return self._generate(content)

    def _generate(self, content: str) -> str:
        evaluation = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_message},
                {"role": "user", "content": content},
            ],
            temperature=0.3,
        )
        return evaluation.choices[0].message.content
