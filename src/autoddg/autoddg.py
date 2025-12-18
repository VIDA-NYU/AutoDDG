from __future__ import annotations

from typing import Any

from beartype import beartype
from pandas import DataFrame

from .description import DatasetDescriptionGenerator, SearchFocusedDescription
from .evaluation import BaseEvaluator
from .llm import LocalLLMClient, OpenAICompatibleClient
from .profiling import SemanticProfiler, profile_dataset
from .topic import DatasetTopicGenerator


@beartype
class AutoDDG:
    """AutoDDG - Automated Dataset Description Generator

    AutoDDG is the library's entry class, exposing:
    * profiling,
    * semantic analysis,
    * topic generation,
    * dataset description,
    * search-focused description expansion, and
    * optional description evaluation

    Args:
        client (Any): OpenAI-compatible client (e.g. ``openai.OpenAI(...)``) or None
            for local LLM.
        model_name (str): Model identifier (e.g. ``"gpt-4o"`` for API or
            ``"Qwen/Qwen2.5-7B-Instruct"`` for local).
        use_local_llm (bool): If True, use local LLM via transformers.
            Requires transformers and torch.
        local_llm_device (str | None): Device for local LLM ("cuda", "cpu", or None
            for auto).
        local_llm_dtype (str | None): Data type for local LLM ("float16", "bfloat16",
            "float32", or None for auto).
        description_temperature (float): Temperature for description generation.
        description_words (int): Target word count for generated descriptions.
        search_model_name (str | None): Override model for search-expansion.
        semantic_model_name (str | None): Override model for semantic profiling.
        topic_temperature (float): Temperature for topic generation.
        evaluator (BaseEvaluator | None): Optional evaluator for quality scoring.

    Examples:
        Basic usage with API:

            >>> import openai
            >>> from autoddg import AutoDDG
            >>> client = openai.OpenAI(api_key="sk-...")
            >>> pipe = AutoDDG(
            ...     client=client, model_name="gpt-4o", description_words=100
            ... )
            >>> sample_csv = (
            ...     "city,country,population\\nLondon,UK,8908081\\nLeeds,UK,789194"
            ... )
            >>> prompt, desc = pipe.describe_dataset(dataset_sample=sample_csv)
            >>> print(desc)

        Basic usage with local LLM:

            >>> from autoddg import AutoDDG
            >>> pipe = AutoDDG(
            ...     client=None,
            ...     model_name="Qwen/Qwen2.5-7B-Instruct",
            ...     use_local_llm=True,
            ...     description_words=100
            ... )
            >>> sample_csv = (
            ...     "city,country,population\\nLondon,UK,8908081\\nLeeds,UK,789194"
            ... )
            >>> prompt, desc = pipe.describe_dataset(dataset_sample=sample_csv)
            >>> print(desc)

        Advanced usage with topic and evaluator:

            >>> import pandas as pd
            >>> from autoddg import GPTEvaluator
            >>> df = pd.DataFrame({
            ...     "city": ["London", "Leeds"],
            ...     "country": ["UK", "UK"],
            ...     "population": [8908081, 789194],
            ... })
            >>> profile, semantic = pipe.profile_dataframe(df)
            >>> topic = pipe.generate_topic("UK Cities", None, df.to_csv(index=False))
            >>> _, desc = pipe.describe_dataset(
            ...     dataset_sample=df.to_csv(index=False),
            ...     dataset_profile=profile,
            ...     use_profile=True,
            ...     semantic_profile=semantic,
            ...     use_semantic_profile=True,
            ...     data_topic=topic, use_topic=True,
            ... )
            >>> evaluator = GPTEvaluator(gpt4_api_key="sk-...")
            >>> pipe.set_evaluator(evaluator)
            >>> scores = pipe.evaluate_description(desc)
            >>> print(scores)
    """

    def __init__(
        self,
        client: Any | None = None,
        model_name: str = "gpt-4o",
        *,
        use_local_llm: bool = False,
        local_llm_device: str | None = None,
        local_llm_dtype: str | None = None,
        description_temperature: float = 0.0,
        description_words: int = 100,
        search_model_name: str | None = None,
        semantic_model_name: str | None = None,
        topic_temperature: float = 0.0,
        evaluator: BaseEvaluator | None = None,
    ) -> None:
        # Initialize LLM client
        if use_local_llm:
            if client is not None:
                raise ValueError(
                    "Cannot specify both client and use_local_llm=True. "
                    "For local LLM, set client=None and use_local_llm=True."
                )
            llm_client = LocalLLMClient(
                model_name=model_name,
                device=local_llm_device,
                torch_dtype=local_llm_dtype,
            )
        elif client is not None:
            llm_client = OpenAICompatibleClient(client)
        else:
            raise ValueError(
                "Must provide either client (for API) or set use_local_llm=True "
                "(for local LLM)"
            )

        self.client = client  # Keep for backward compatibility
        self.model_name = model_name
        self.description_generator = DatasetDescriptionGenerator(
            client=llm_client,
            model_name=model_name,
            temperature=description_temperature,
            description_words=description_words,
        )
        self.semantic_profiler = SemanticProfiler(
            client=llm_client,
            model_name=semantic_model_name or model_name,
        )
        self.topic_generator = DatasetTopicGenerator(
            client=llm_client,
            model_name=model_name,
            temperature=topic_temperature,
        )
        self.search_description = SearchFocusedDescription(
            client=llm_client,
            model_name=search_model_name or model_name,
        )
        self.evaluator = evaluator

    def describe_dataset(
        self,
        dataset_sample: str,
        dataset_profile: str | None = None,
        use_profile: bool = False,
        semantic_profile: str | None = None,
        use_semantic_profile: bool = False,
        data_topic: str | None = None,
        use_topic: bool = False,
    ) -> tuple[str, str]:
        """
        Produce a short description from a CSV sample with optional context

        Args:
            dataset_sample: CSV text containing example rows
            dataset_profile: Structural profile text
            use_profile: Include the structural profile if True
            semantic_profile: Natural-language column semantics
            use_semantic_profile: Include the semantic profile if True
            data_topic: Short topic string for the dataset
            use_topic: Include the topic if True

        Returns:
            (prompt, description)
        """

        return self.description_generator.generate_description(
            dataset_sample=dataset_sample,
            dataset_profile=dataset_profile,
            use_profile=use_profile,
            semantic_profile=semantic_profile,
            use_semantic_profile=use_semantic_profile,
            data_topic=data_topic,
            use_topic=use_topic,
        )

    def profile_dataframe(self, dataframe: DataFrame) -> tuple[str, str]:
        """
        Summarise structure and coverage using the datamart profiler

        Ref: https://pypi.org/project/datamart-profiler/

        Args:
            dataframe: Input frame

        Returns:
            (profile_text, semantic_notes)
        """

        return profile_dataset(dataframe)

    def analyze_semantics(
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
        Infer column semantics with an LLM and return a short overview

        Processing modes available:
        1. Sequential mode (default): Processes columns one by one sequentially.
        2. Multi-threaded mode (use_multi_threading=True): Uses multi-threading to
           process columns in parallel for faster execution. Only available for
           OpenAI API clients.
        3. Group mode (use_group_prompting=True): Processes columns in groups via
           group prompting, reducing API calls.
           - If group_size=0: Processes all columns in a single prompt (most
             efficient).
           - If group_size>0: Processes columns in groups of group_size.
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
            Summary of column semantics
        """

        return self.semantic_profiler.analyze_dataframe(
            dataframe,
            use_group_prompting=use_group_prompting,
            use_multi_threading=use_multi_threading,
            use_batch_processing=use_batch_processing,
            max_workers=max_workers,
            group_size=group_size,
            batch_size=batch_size,
        )

    def generate_topic(
        self, title: str, original_description: str | None, dataset_sample: str
    ) -> str:
        """
        Generate a 2–3 word topic from title description and sample

        Args:
            title: Dataset title
            original_description: Existing description if available
            dataset_sample: CSV text sample

        Returns:
            Short topic string
        """

        return self.topic_generator.generate_topic(
            title, original_description, dataset_sample
        )

    def expand_description_for_search(
        self, description: str, topic: str
    ) -> tuple[str, str]:
        """
        Expand a readable description into a search-oriented variant

        Args:
            description: Original dataset description
            topic: Topic string

        Returns:
            (prompt, expanded_description)
        """

        return self.search_description.expand_description(description, topic)

    def evaluate_description(self, description: str) -> str:
        """
        Score a description with the configured evaluator

        Args:
            description: Description text to score

        Returns:
            Evaluation response

        Raises:
            RuntimeError: If no evaluator is set
        """

        if self.evaluator is None:
            raise RuntimeError(
                "No evaluator configured for AutoDDG. Provide one via set_evaluator()."
            )
        return self.evaluator.evaluate(description)

    def set_evaluator(self, evaluator: BaseEvaluator) -> None:
        """
        Attach or replace the evaluator to use for scoring

        Args:
            evaluator: Evaluator instance
        """

        self.evaluator = evaluator
