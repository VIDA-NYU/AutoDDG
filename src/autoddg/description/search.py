from __future__ import annotations

from typing import Any, Tuple

from beartype import beartype

from ..utils import load_prompts


@beartype
class SearchFocusedDescription:
    """Create search-optimised dataset descriptions

    Expand a human-readable description into a search-optimised outline
    """

    def __init__(self, client: Any, model_name: str = "gpt-4o-mini") -> None:
        self.client = client
        self.model = model_name
        prompts = load_prompts()["search_expansion"]
        self._template = prompts["template"].strip()
        self._system_message = prompts["system_message"].strip()
        self._user_prompt = prompts["user_prompt"]

    def expand_description(self, initial_description: str, topic: str) -> Tuple[str, str]:
        """Expand an initial dataset description for retrieval tasks

        Args:
            initial_description: Baseline description that will be enriched
            topic: Topic context anchoring the search-optimised output

        Returns:
            (prompt, expanded_description)
        """

        prompt = self._user_prompt.format(
            topic=topic,
            initial_description=initial_description,
            template=self._template,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_message},
                {"role": "user", "content": prompt},
            ],
        )

        expanded_description = response.choices[0].message.content
        return prompt, expanded_description
