from __future__ import annotations

from typing import Any, Tuple

from beartype import beartype
from pandas import DataFrame

from .description import DatasetDescriptionGenerator, SearchFocusedDescription
from .evaluation import BaseEvaluator
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
        client (Any): OpenAI-compatible client (e.g. ``openai.OpenAI(...)``).
        model_name (str): Default model identifier (e.g. ``"gpt-4o"``).
        description_temperature (float): Temperature for description generation.
        description_words (int): Target word count for generated descriptions.
        search_model_name (str | None): Override model for search-expansion.
        semantic_model_name (str | None): Override model for semantic profiling.
        topic_temperature (float): Temperature for topic generation.
        evaluator (BaseEvaluator | None): Optional evaluator for quality scoring.

    Examples:
        Basic usage:

            >>> import openai
            >>> from autoddg import AutoDDG
            >>> client = openai.OpenAI(api_key="sk-...")
            >>> pipe = AutoDDG(client=client, model_name="gpt-4o", description_words=100)
            >>> sample_csv = "city,country,population\\nLondon,UK,8908081\\nLeeds,UK,789194"
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
        client: Any,
        model_name: str,
        *,
        description_temperature: float = 0.0,
        description_words: int = 100,
        search_model_name: str | None = None,
        semantic_model_name: str | None = None,
        topic_temperature: float = 0.0,
        evaluator: BaseEvaluator | None = None,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.description_generator = DatasetDescriptionGenerator(
            client=client,
            model_name=model_name,
            temperature=description_temperature,
            description_words=description_words,
        )
        self.semantic_profiler = SemanticProfiler(
            client=client,
            model_name=semantic_model_name or model_name,
        )
        self.topic_generator = DatasetTopicGenerator(
            client=client,
            model_name=model_name,
            temperature=topic_temperature,
        )
        self.search_description = SearchFocusedDescription(
            client=client,
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
    ) -> Tuple[str, str]:
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

    def profile_dataframe(self, dataframe: DataFrame) -> Tuple[str, str]:
        """
        Summarise structure and coverage using the datamart profiler

        Ref: https://pypi.org/project/datamart-profiler/

        Args:
            dataframe: Input frame

        Returns:
            (profile_text, semantic_notes)
        """

        return profile_dataset(dataframe)

    def analyze_semantics(self, dataframe: DataFrame) -> str:
        """
        Infer column semantics with an LLM and return a short overview

        Args:
            dataframe: Input frame

        Returns:
            Summary of column semantics
        """

        return self.semantic_profiler.analyze_dataframe(dataframe)

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

        return self.topic_generator.generate_topic(title, original_description, dataset_sample)

    def expand_description_for_search(self, description: str, topic: str) -> Tuple[str, str]:
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
