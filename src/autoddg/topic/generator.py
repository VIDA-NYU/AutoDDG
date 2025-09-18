from __future__ import annotations

from typing import Any

from beartype import beartype

from ..utils import load_prompts


@beartype
class DatasetTopicGenerator:
    """
    Generate a short topic few words describing topic of a dataset
    """

    def __init__(self, client: Any, model_name: str, temperature: float = 0.0) -> None:
        self.client = client
        self.model = model_name
        self.temperature = float(temperature)
        prompts = load_prompts()["topic_generation"]
        self._system_message = prompts["system_message"].strip()
        self._user_prompt = prompts["user_prompt"]

    def _build_prompt(
        self, title: str, original_description: str | None, dataset_sample: str
    ) -> str:
        description_block = (
            f"Original Description: {original_description}\n" if original_description else ""
        )
        return self._user_prompt.format(
            title=title,
            description_block=description_block,
            dataset_sample=dataset_sample,
        )

    def generate_topic(
        self, title: str, original_description: str | None, dataset_sample: str
    ) -> str:
        """
        Return a trimmed topic string for a dataset

        Args:
            title: Dataset title
            original_description: Existing description if available
            dataset_sample: CSV text sample

        Returns:
            Topic string
        """

        prompt = self._build_prompt(title, original_description, dataset_sample)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()
