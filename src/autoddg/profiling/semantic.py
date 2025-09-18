from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List

from beartype import beartype
from pandas import DataFrame

from ..utils import load_prompts


@beartype
class SemanticProfiler:
    """Infer semantic information for each column using an LLM"""

    def __init__(self, client: Any, model_name: str = "gpt-4o-mini") -> None:
        self.client = client
        self.model = model_name
        prompts = load_prompts()["semantic_profiler"]
        self._template = prompts["template"]
        self._response_example = prompts["response_example"]
        self._system_message = prompts["system_message"].strip()
        self._user_prompt = prompts["user_prompt"]

    def _fix_json_response(self, response_text: str) -> str:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            return response_text

        response_body = match.group()
        open_braces = response_body.count("{")
        close_braces = response_body.count("}")
        response_body += "}" * (open_braces - close_braces)
        response_body = re.sub(r",\s*}", "}", response_body)
        return response_body

    def _build_prompt(self, column_name: str, sample_values: Iterable[str]) -> str:
        sample_text = ", ".join(sample_values)
        return self._user_prompt.format(
            template=self._template,
            response_example=self._response_example,
            column_name=column_name,
            sample_values=sample_text,
        )

    def get_semantic_type(
        self, column_name: str, sample_values: Iterable[str]
    ) -> Dict[str, Any] | None:
        """
        Return parsed semantic metadata for a column or None on parse failure

        Args:
            column_name: Column name
            sample_values: Example values

        Returns:
            Semantic metadata dict or None
        """

        prompt = self._build_prompt(column_name, sample_values)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_message},
                {"role": "user", "content": prompt},
            ],
        )
        response_text = response.choices[0].message.content
        response_text = self._fix_json_response(response_text)

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return None

    def analyze_dataframe(self, dataframe: DataFrame) -> str:
        """
        Summarise detected semantics per column in plain English

        Args:
            dataframe: Input frame

        Returns:
            Text summary of semantics
        """

        def _get_sample(data_pd: DataFrame, sample_size: int) -> DataFrame:
            if sample_size < len(data_pd):
                return data_pd.sample(sample_size, random_state=9)
            return data_pd

        semantic_summary: List[str] = []
        dataframe_sample = _get_sample(dataframe, 5)

        for column in dataframe.columns:
            sample_values = dataframe_sample[column].astype(str).tolist()
            semantic_description: Dict[str, Any] | None = None
            retry_count = 0
            while semantic_description is None and retry_count < 3:
                semantic_description = self.get_semantic_type(column, sample_values)
                retry_count += 1
            if semantic_description is None:
                continue

            column_summary = f"**{column}**: "
            entity_type = semantic_description.get("Entity Type", "Unknown")
            if entity_type and entity_type.lower() not in {"", "unknown"}:
                column_summary += f"Represents {entity_type.lower()}. "

            temporal = semantic_description.get("Temporal", {})
            if temporal.get("isTemporal"):
                resolution = temporal.get("resolution", "unknown")
                column_summary += f"Contains temporal data (resolution: {resolution}). "

            spatial = semantic_description.get("Spatial", {})
            if spatial.get("isSpatial"):
                resolution = spatial.get("resolution", "unknown")
                column_summary += f"Contains spatial data (resolution: {resolution}). "

            domain_type = semantic_description.get("Domain-Specific Types", "Unknown")
            if domain_type and domain_type.lower() not in {"", "unknown"}:
                column_summary += f"Domain-specific type: {domain_type.lower()}. "

            function_context = semantic_description.get("Function/Usage Context", "Unknown")
            if function_context and function_context.lower() not in {"", "unknown"}:
                column_summary += f"Function/Usage context: {function_context.lower()}. "

            semantic_summary.append(column_summary)

        final_summary = "The key semantic information for this dataset includes:\n" + "\n".join(
            semantic_summary
        )
        return final_summary
