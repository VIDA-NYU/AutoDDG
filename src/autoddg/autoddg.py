from __future__ import annotations

from typing import Any, Mapping, Tuple

from beartype import beartype
from pandas import DataFrame

from .description import DatasetDescriptionGenerator, SearchFocusedDescription
from .evaluation import BaseEvaluator
from .profiling import SemanticProfiler, profile_dataset
from .topic import DatasetTopicGenerator
from .utils import LLMClientFactory


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

    The class can be configured in two stages. First initialise `AutoDDG`
    with your preferred temperatures, word-count targets, or evaluator. Then
    bind a provider by calling `with_provider`, which will either
    construct a LiteLLM-backed client for you or attach an existing compatible
    client. Ref: https://github.com/BerriAI/litellm.

    Args:
        description_temperature (float): Temperature for description generation.
        description_words (int): Target word count for generated descriptions.
        search_model_name (str | None): Override model for search-expansion.
        semantic_model_name (str | None): Override model for semantic profiling.
        topic_temperature (float): Temperature for topic generation.
        evaluator (BaseEvaluator | None): Optional evaluator for quality scoring.
        client (Any | None): Optional OpenAI-compatible client used to seed the instance immediately.
        model_name (str | None): Default model identifier to pair with ``client``.

    Examples:
        Basic usage:

            >>> from autoddg import AutoDDG
            >>> pipe = AutoDDG(description_words=100).with_provider(
            ...     provider="openai", model_name="gpt-4o"
            ... )
            >>> sample_csv = "city,country,population\\nLondon,UK,8908081\\nLeeds,UK,789194"
            >>> prompt, desc = pipe.describe_dataset(dataset_sample=sample_csv)
            >>> print(desc)

        Advanced usage with topic and evaluator:

            >>> import pandas as pd
            >>> from autoddg import GPTEvaluator
            >>> base = AutoDDG(topic_temperature=0.2)
            >>> pipe = base.with_provider(provider="openai", model_name="gpt-4o")
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

    _shared_factory: LLMClientFactory | None = None

    def __init__(
        self,
        *,
        description_temperature: float = 0.0,
        description_words: int = 100,
        search_model_name: str | None = None,
        semantic_model_name: str | None = None,
        topic_temperature: float = 0.0,
        evaluator: BaseEvaluator | None = None,
        client: Any | None = None,
        model_name: str | None = None,
    ) -> None:
        self._description_temperature = description_temperature
        self._description_words = description_words
        self._search_model_name = search_model_name
        self._semantic_model_name = semantic_model_name
        self._topic_temperature = topic_temperature

        self.client = None
        self.model_name = None
        self.description_generator: DatasetDescriptionGenerator | None = None
        self.semantic_profiler: SemanticProfiler | None = None
        self.topic_generator: DatasetTopicGenerator | None = None
        self.search_description: SearchFocusedDescription | None = None
        self.evaluator = evaluator

        if (client is None) ^ (model_name is None):
            raise ValueError(
                "Both `client` and `model_name` must be provided together when seeding AutoDDG."
            )
        if client is not None and model_name is not None:
            self._bind_client(client, model_name)

    def _bind_client(self, client: Any, model_name: str) -> None:
        self.client = client
        self.model_name = model_name
        self.description_generator = DatasetDescriptionGenerator(
            client=client,
            model_name=model_name,
            temperature=self._description_temperature,
            description_words=self._description_words,
        )
        self.semantic_profiler = SemanticProfiler(
            client=client,
            model_name=self._semantic_model_name or model_name,
        )
        self.topic_generator = DatasetTopicGenerator(
            client=client,
            model_name=model_name,
            temperature=self._topic_temperature,
        )
        self.search_description = SearchFocusedDescription(
            client=client,
            model_name=self._search_model_name or model_name,
        )

    def _ensure_ready(self) -> None:
        if self.client is None or self.description_generator is None:
            raise RuntimeError("AutoDDG is not bound to a provider. Call `with_provider()` first.")

    @classmethod
    @beartype
    def _get_factory(cls, override: LLMClientFactory | None = None) -> LLMClientFactory:
        if override is not None:
            return override
        if cls._shared_factory is None:
            cls._shared_factory = LLMClientFactory()
        return cls._shared_factory

    @classmethod
    @beartype
    def list_providers(
        cls,
        *,
        include_aliases: bool = False,
        factory: LLMClientFactory | None = None,
    ) -> Tuple[str, ...]:
        """Expose the configured provider identifiers from the shared factory."""

        llm_factory = cls._get_factory(factory)
        return llm_factory.list_providers(include_aliases=include_aliases)

    @classmethod
    @beartype
    def list_model_names(
        cls,
        provider: str,
        *,
        factory: LLMClientFactory | None = None,
    ) -> Tuple[str, ...]:
        """Return available model names for ``provider`` from the shared factory."""

        llm_factory = cls._get_factory(factory)
        return llm_factory.list_model_names(provider)

    @classmethod
    @beartype
    def describe_provider(
        cls,
        provider: str,
        *,
        factory: LLMClientFactory | None = None,
    ) -> Mapping[str, Any]:
        """Return metadata for ``provider`` from the shared factory."""

        llm_factory = cls._get_factory(factory)
        return llm_factory.describe_provider(provider)

    @beartype
    def with_provider(
        self,
        *,
        provider: str | None = None,
        model_name: str,
        api_key: str | None = None,
        client: Any | None = None,
        factory: LLMClientFactory | None = None,
        factory_options: Mapping[str, Any] | None = None,
    ) -> "AutoDDG":
        """Return a new `AutoDDG` bound to the requested provider.

        Args:
            provider: Identifier of the LLM provider (e.g. ``"openai"``). Required
                when ``client`` is not supplied.
            model_name: Default model identifier to use with the provider/client.
            api_key: Optional API key overriding environment variables.
            client: Existing OpenAI-compatible client. When provided, ``provider``
                is ignored.
            factory: Optional `LLMClientFactory` instance.
            factory_options: Extra keyword arguments forwarded to
                `LLMClientFactory.create`.

        Returns:
            Configured `AutoDDG` instance.
        """

        if client is None:
            if provider is None:
                raise ValueError(
                    "Either `provider` or `client` must be supplied when binding AutoDDG to a provider."
                )
            llm_factory = self._get_factory(factory)
            options = dict(factory_options or {})
            try:
                client = llm_factory.create(
                    provider,
                    api_key=api_key,
                    default_model=model_name,
                    **options,
                )
            except ValueError as exc:
                raise ValueError(f"Failed to configure provider '{provider}': {exc}") from exc

        clone = AutoDDG(
            description_temperature=self._description_temperature,
            description_words=self._description_words,
            search_model_name=self._search_model_name,
            semantic_model_name=self._semantic_model_name,
            topic_temperature=self._topic_temperature,
            evaluator=self.evaluator,
            client=client,
            model_name=model_name,
        )
        return clone

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

        self._ensure_ready()
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

        self._ensure_ready()
        return profile_dataset(dataframe)

    def analyze_semantics(self, dataframe: DataFrame) -> str:
        """
        Infer column semantics with an LLM and return a short overview

        Args:
            dataframe: Input frame

        Returns:
            Summary of column semantics
        """

        self._ensure_ready()
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

        self._ensure_ready()
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

        self._ensure_ready()
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
