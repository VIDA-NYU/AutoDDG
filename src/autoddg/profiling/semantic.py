from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from beartype import beartype
from pandas import DataFrame

from ..llm import LLMClient, LocalLLMClient
from ..utils import load_prompts

logger = logging.getLogger(__name__)


@beartype
class SemanticProfiler:
    """Infer semantic information for each column using an LLM"""

    def __init__(self, client: LLMClient | Any, model_name: str = "gpt-4o-mini") -> None:
        # Support both LLMClient and legacy OpenAI clients
        if isinstance(client, LLMClient):
            self.llm_client = client
        else:
            # Legacy support: wrap OpenAI-compatible client
            from ..llm import OpenAICompatibleClient

            self.llm_client = OpenAICompatibleClient(client)
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
    ) -> dict[str, Any] | None:
        """
        Return parsed semantic metadata for a column or None on parse failure

        Args:
            column_name: Column name
            sample_values: Example values

        Returns:
            Semantic metadata dict or None
        """

        prompt = self._build_prompt(column_name, sample_values)
        response = self.llm_client.chat_completions_create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_message},
                {"role": "user", "content": prompt},
            ],
        )
        response_text = response["choices"][0]["message"]["content"]
        response_text = self._fix_json_response(response_text)

        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            return None
        # The model may return valid JSON that is not the expected object
        # (e.g. a bare string or list); treat that as a failure so callers retry.
        return parsed if isinstance(parsed, dict) else None

    def _get_semantic_types_group(
        self, column_data: list[tuple[str, list[str]]]
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """
        Get semantic types for multiple columns in a single group prompt.

        Args:
            column_data: List of tuples (column_name, sample_values_list).

        Returns:
            Tuple of (results_dict, stats_dict) where:
            - results_dict maps column_name -> semantic_description
            - stats_dict contains runtime and token usage statistics
        """
        prompts = load_prompts()["semantic_profiler"]
        group_prompt_template = prompts["group_user_prompt"]
        group_response_example = prompts["group_response_example"]

        # Build group prompt with all columns
        columns_info = []
        for column_name, sample_values in column_data:
            columns_info.append(f"Column: {column_name}\nSample values: {sample_values}")

        columns_text = "\n\n".join(columns_info)

        prompt = group_prompt_template.format(
            template=self._template,
            response_example=group_response_example,
            columns_text=columns_text,
        )

        try:
            response = self.llm_client.chat_completions_create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant skilled in dataset semantic "
                            "analysis. You analyze multiple columns efficiently in a "
                            "single response."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            response_text = response["choices"][0]["message"]["content"]

            # Fix JSON if needed
            try:
                response_text = self._fix_json_response(response_text)
                group_results = json.loads(response_text)
            except (json.JSONDecodeError, AttributeError) as e:
                return {}, {"error": str(e)}

            # Validate and extract results. Only accept dict-shaped values:
            # anything else (bare string, list, ...) is a malformed answer for
            # that column and is left out so the caller can retry it.
            results = {}
            if isinstance(group_results, dict):
                for column_name, _ in column_data:
                    value = group_results.get(column_name)
                    if isinstance(value, dict):
                        results[column_name] = value

            # Record statistics
            stats = {
                "input_tokens": (
                    response.usage.prompt_tokens
                    if hasattr(response, "usage") and response.usage
                    else 0
                ),
                "output_tokens": (
                    response.usage.completion_tokens
                    if hasattr(response, "usage") and response.usage
                    else 0
                ),
                "total_tokens": (
                    response.usage.total_tokens
                    if hasattr(response, "usage") and response.usage
                    else 0
                ),
                "num_columns": len(column_data),
            }

            return results, stats

        except Exception as e:
            return {}, {"error": str(e), "num_columns": len(column_data)}

    def _process_single_column(
        self, column_name: str, sample_values: list[str]
    ) -> tuple[str, dict[str, Any] | None]:
        """
        Process a single column to get semantic type with retry logic.

        Args:
            column_name: Column name.
            sample_values: Sample values from the column.

        Returns:
            Tuple of (column_name, semantic_description) or (column_name, None)
            if failed.
        """
        semantic_description: dict[str, Any] | None = None
        retry_count = 0
        while semantic_description is None and retry_count < 3:
            semantic_description = self.get_semantic_type(column_name, sample_values)
            retry_count += 1

        return (column_name, semantic_description)

    @staticmethod
    def _is_affirmative(value: Any) -> bool:
        """True only for boolean True or yes/true strings (the model sometimes
        answers "Yes"/"No" instead of the requested JSON booleans)."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"yes", "true"}
        return False

    def _create_column_summary(self, column: str, semantic_description: dict[str, Any]) -> str:
        """
        Create a formatted summary string for a column's semantic description.

        Fields whose values do not have the expected type are skipped rather
        than crashing the whole summary: the model occasionally drifts from the
        requested schema (e.g. "Temporal": "No" or a list of entity types).

        Args:
            column: Column name
            semantic_description: Semantic metadata dictionary

        Returns:
            Formatted summary string
        """
        column_summary = f"**{column}**: "
        entity_type = semantic_description.get("Entity Type", "Unknown")
        if isinstance(entity_type, str) and entity_type.lower() not in {"", "unknown"}:
            column_summary += f"Represents {entity_type.lower()}. "

        temporal = semantic_description.get("Temporal", {})
        if isinstance(temporal, dict) and self._is_affirmative(temporal.get("isTemporal")):
            resolution = temporal.get("resolution", "unknown")
            column_summary += f"Contains temporal data (resolution: {resolution}). "

        spatial = semantic_description.get("Spatial", {})
        if isinstance(spatial, dict) and self._is_affirmative(spatial.get("isSpatial")):
            resolution = spatial.get("resolution", "unknown")
            column_summary += f"Contains spatial data (resolution: {resolution}). "

        domain_type = semantic_description.get("Domain-Specific Types", "Unknown")
        if isinstance(domain_type, str) and domain_type.lower() not in {"", "unknown"}:
            column_summary += f"Domain-specific type: {domain_type.lower()}. "

        function_context = semantic_description.get("Function/Usage Context", "Unknown")
        if isinstance(function_context, str) and function_context.lower() not in {"", "unknown"}:
            column_summary += f"Function/Usage context: {function_context.lower()}. "

        return column_summary

    def analyze_dataframe(
        self,
        dataframe: DataFrame,
        *,
        use_group_prompting: bool = False,
        use_multi_threading: bool = False,
        use_batch_processing: bool = False,
        max_workers: int | None = None,
        group_size: int = 0,
        batch_size: int = 4,
    ) -> str:
        """
        Summarise detected semantics per column in plain English.

        Processing modes available:
        1. Sequential mode (default): Processes columns one by one sequentially.
        2. Multi-threaded mode (use_multi_threading=True): Uses multi-threading to
           process columns in parallel for faster execution. Only available for
           OpenAI API clients.
        3. Group mode (use_group_prompting=True): Processes columns in groups via
           group prompting, reducing API calls.
           - If group_size=0: Processes all columns in a single prompt (most
             efficient).
           - If group_size>0: Processes columns in groups of that size.
        4. Batch mode (use_batch_processing=True): Processes columns in batches
           using batch inference. Only available for local LLMs. Takes precedence
           over other modes.

        Args:
            dataframe: Input frame
            use_group_prompting: If True, use group prompting (single API call for all
                columns or groups). Takes precedence over use_multi_threading.
            use_multi_threading: If True, use multi-threading for individual column
                processing (only used if use_group_prompting=False and
                use_batch_processing=False). Only works with OpenAI API clients.
            use_batch_processing: If True, use batch processing for local LLMs.
                Only works with LocalLLMClient. Takes precedence over other modes.
            max_workers: Maximum number of concurrent workers for multi-threaded mode.
                Default: min(32, num_columns).
            group_size: Group size for group prompting. If 0, process all columns
                at once. If >0, process in groups of that size.
            batch_size: Batch size for batch processing mode. Only used with local LLMs.

        Returns:
            Text summary of semantics
        """

        def _get_sample(data_pd: DataFrame, sample_size: int) -> DataFrame:
            if sample_size < len(data_pd):
                return data_pd.sample(sample_size, random_state=9)
            return data_pd

        semantic_summary: list[str] = []
        dataframe_sample = _get_sample(dataframe, 5)

        # Prepare column data
        column_data: list[tuple[str, list[str]]] = []
        for column in dataframe.columns:
            # pandas >= 2 astype(str) keeps missing values as float NaN, which
            # violates the list[str] type the profiling methods expect (and lets
            # raw NaN leak into prompts). Blank them out before stringifying.
            sample_values = dataframe_sample[column].fillna("").astype(str).tolist()
            column_data.append((column, sample_values))

        num_columns = len(column_data)
        results: dict[str, dict[str, Any]] = {}

        # Check if using local LLM
        is_local_llm = isinstance(self.llm_client, LocalLLMClient)

        # Batch processing mode (only for local LLMs)
        if use_batch_processing:
            if not is_local_llm:
                raise ValueError(
                    "Batch processing is only available for local LLMs. "
                    "Use use_local_llm=True when initializing AutoDDG."
                )
            if not hasattr(self.llm_client, "chat_completions_create_batch"):
                raise ValueError(
                    "Local LLM client does not support batch processing. "
                    "Please update to a version that supports batch processing."
                )

            # Process columns in batches
            for i in range(0, len(column_data), batch_size):
                batch = column_data[i : i + batch_size]
                batch_messages = []
                batch_columns = []

                for column_name, sample_values in batch:
                    prompt = self._build_prompt(column_name, sample_values)
                    batch_messages.append(
                        [
                            {"role": "system", "content": self._system_message},
                            {"role": "user", "content": prompt},
                        ]
                    )
                    batch_columns.append(column_name)

                # Process batch
                batch_responses = self.llm_client.chat_completions_create_batch(
                    model=self.model,
                    messages_list=batch_messages,
                )

                # Parse batch results
                for column_name, response in zip(batch_columns, batch_responses, strict=True):
                    response_text = response["choices"][0]["message"]["content"]
                    response_text = self._fix_json_response(response_text)
                    try:
                        semantic_description = json.loads(response_text)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(semantic_description, dict):
                        results[column_name] = semantic_description

        elif use_group_prompting:
            # Group mode: process columns in groups or all at once
            if group_size == 0:
                groups = [column_data]
            else:
                groups = [
                    column_data[i : i + group_size] for i in range(0, len(column_data), group_size)
                ]
            for group in groups:
                # Match the sequential mode's 3-attempt policy: a failed or
                # partially answered group call is retried with just the
                # still-missing columns instead of failing silently.
                pending = group
                last_error: Any = None
                for _attempt in range(3):
                    group_results, stats = self._get_semantic_types_group(pending)
                    results.update(group_results)
                    last_error = stats.get("error", last_error)
                    pending = [item for item in pending if item[0] not in results]
                    if not pending:
                        break
                if pending:
                    logger.warning(
                        "Semantic profiling produced no usable result for %d "
                        "column(s) after 3 attempts (last error: %s): %s",
                        len(pending),
                        last_error,
                        ", ".join(item[0] for item in pending),
                    )

        elif use_multi_threading:
            # Multi-threaded mode: process columns in parallel (only for OpenAI API)
            if is_local_llm:
                raise ValueError(
                    "Multi-threading is only available for OpenAI API clients. "
                    "For local LLMs, use use_batch_processing=True instead."
                )
            if max_workers is None:
                max_workers = min(32, num_columns)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_column = {
                    executor.submit(self._process_single_column, column, sample_values): column
                    for column, sample_values in column_data
                }

                # Collect results as they complete
                for future in as_completed(future_to_column):
                    try:
                        col_name, semantic_description = future.result()
                        if semantic_description is not None:
                            results[col_name] = semantic_description
                    except Exception:
                        # Skip failed columns
                        pass

        else:
            # Sequential mode: process columns one by one
            for column, sample_values in column_data:
                semantic_description: dict[str, Any] | None = None
                retry_count = 0
                while semantic_description is None and retry_count < 3:
                    semantic_description = self.get_semantic_type(column, sample_values)
                    retry_count += 1
                if semantic_description is not None:
                    results[column] = semantic_description

        # Create summaries from results
        for column in dataframe.columns:
            if column in results:
                column_summary = self._create_column_summary(column, results[column])
                semantic_summary.append(column_summary)

        final_summary = "The key semantic information for this dataset includes:\n" + "\n".join(
            semantic_summary
        )
        return final_summary
