from __future__ import annotations

import importlib.resources as resources
import os
from collections import defaultdict
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping

import litellm
import yaml
from beartype import beartype
from litellm import completion, get_llm_provider

if hasattr(litellm, "suppress_debug_info"):
    litellm.suppress_debug_info = True
from thefuzz import process


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration container for a provider supported by `LLMClientFactory`."""

    api_key_env: tuple[str, ...] = field(default_factory=tuple)
    base_url: str | None = None
    extra_options: Mapping[str, Any] = field(default_factory=dict)


def _load_provider_configuration() -> tuple[dict[str, ProviderConfig], dict[str, str]]:
    """Load provider defaults and aliases from ``provider_defaults.yaml``."""

    try:
        config_path = resources.files(__package__).joinpath("provider_defaults.yaml")
    except (FileNotFoundError, ModuleNotFoundError):
        return {}, {}

    if not config_path.is_file():
        return {}, {}

    with config_path.open("rb") as handle:
        try:
            raw_config = yaml.safe_load(handle) or {}
        except yaml.YAMLError:
            return {}, {}

    providers_section = raw_config.get("providers", {}) or {}
    provider_configs: dict[str, ProviderConfig] = {}
    alias_map: dict[str, str] = {}

    for name, config in providers_section.items():
        if not isinstance(config, Mapping):
            continue

        key = str(name).lower()
        api_key_env = tuple(str(env) for env in config.get("api_key_env", ()) or ())
        base_url = config.get("base_url")
        extra_options = config.get("extra_options") or {}

        provider_configs[key] = ProviderConfig(
            api_key_env=api_key_env,
            base_url=str(base_url) if base_url else None,
            extra_options=dict(extra_options),
        )

        alias_map[key] = key
        for alias in config.get("aliases", ()) or ():
            alias_map[str(alias).lower()] = key

    for alias, target in (raw_config.get("aliases") or {}).items():
        alias_map[str(alias).lower()] = str(target).lower()

    return provider_configs, alias_map


_DEFAULT_PROVIDER_CONFIGS, _DEFAULT_ALIASES = _load_provider_configuration()


def _build_provider_model_index() -> dict[str, tuple[str, ...]]:
    """Build an index mapping provider identifiers to known model names."""

    provider_models: dict[str, set[str]] = defaultdict(set)

    for attribute in dir(litellm):
        if not attribute.endswith("_models"):
            continue

        value = getattr(litellm, attribute)
        if not isinstance(value, (list, tuple, set)):
            continue

        for candidate in value:
            if not isinstance(candidate, str):
                continue

            try:
                _, provider_name, _, _ = get_llm_provider(candidate)
            except Exception:
                continue

            provider_models[provider_name.lower()].add(candidate)

    return {key: tuple(sorted(models)) for key, models in provider_models.items()}


_PROVIDER_MODEL_INDEX = _build_provider_model_index()
_SUGGESTION_SCORE_CUTOFF = 90


class _LiteLLMChatCompletions:
    """Lightweight adapter that exposes a ``create`` method like ``openai``."""

    def __init__(self, default_params: Mapping[str, Any]) -> None:
        self._default_params: Dict[str, Any] = dict(default_params)

    def create(self, **kwargs: Any) -> Any:
        params: Dict[str, Any] = {**self._default_params, **kwargs}
        if "model" not in params:
            raise ValueError(
                "A `model` identifier must be supplied when calling the chat completion API."
            )
        return completion(**params)


class LiteLLMClient:
    """Simple namespace exposing ``chat.completions.create`` backed by LiteLLM."""

    def __init__(self, default_params: Mapping[str, Any]) -> None:
        completions = _LiteLLMChatCompletions(default_params)
        self.chat = SimpleNamespace(completions=completions)


class LLMClientFactory:
    """Factory for building chat-completion clients across multiple providers.

    The factory creates `LiteLLMClient` instances configured with sensible
    defaults for a given provider. Default provider metadata is loaded from
    ``provider_defaults.yaml`` next to this module. Each client exposes the subset
    of the OpenAI Python SDK interface that AutoDDG relies on, namely
    ``chat.completions.create``.
    """

    def __init__(
        self,
        *,
        provider_defaults: Mapping[str, ProviderConfig] | None = None,
        aliases: Mapping[str, str] | None = None,
    ) -> None:
        self._providers: Dict[str, ProviderConfig] = dict(_DEFAULT_PROVIDER_CONFIGS)
        self._aliases: Dict[str, str] = dict(_DEFAULT_ALIASES)

        if provider_defaults:
            for name, config in provider_defaults.items():
                key = name.lower()
                self._providers[key] = config
                self._aliases.setdefault(key, key)

        if aliases:
            for alias, target in aliases.items():
                self._aliases[alias.lower()] = target.lower()

    def _suggest_providers(self, query: str) -> tuple[tuple[str, int], ...]:
        suggestions = process.extractBests(
            query,
            tuple(self._aliases.keys()),
            limit=5,
            score_cutoff=_SUGGESTION_SCORE_CUTOFF,
        )
        best_scores: dict[str, tuple[str, int]] = {}
        for alias, score in suggestions:
            canonical = self._aliases.get(alias)
            if not canonical:
                continue
            current = best_scores.get(canonical)
            if current is None or score > current[1]:
                best_scores[canonical] = (canonical, int(score))
        return tuple(sorted(best_scores.values(), key=lambda item: item[1], reverse=True))

    def _normalise_provider(self, provider: str) -> str:
        key = provider.lower()
        direct = self._aliases.get(key)
        if direct:
            return direct

        suggestions = self._suggest_providers(key)
        if suggestions:
            suggestion_text = ", ".join(f"{name} ({score}%)" for name, score in suggestions)
            raise ValueError("Unknown provider " f"'{provider}'. Did you mean: {suggestion_text}?")

        known = ", ".join(sorted(self._providers.keys()))
        raise ValueError(f"Unknown provider '{provider}'. Known providers: {known or 'N/A'}.")

    def _normalise_model_name(self, provider_key: str, model_name: str) -> str:
        potential_providers = {provider_key}

        config = self._providers.get(provider_key)
        if config and isinstance(config.extra_options, Mapping):
            custom_provider = config.extra_options.get("custom_llm_provider")
            if isinstance(custom_provider, str):
                potential_providers.add(custom_provider.lower())

        potential_providers.update(
            alias for alias, target in self._aliases.items() if target == provider_key
        )

        for candidate_provider in potential_providers:
            models = _PROVIDER_MODEL_INDEX.get(candidate_provider)
            if not models:
                continue

            if model_name in models:
                return model_name

            matches = process.extractBests(
                model_name,
                models,
                limit=5,
                score_cutoff=_SUGGESTION_SCORE_CUTOFF,
            )
            if matches:
                suggestion_text = ", ".join(
                    f"{candidate[0]} ({int(candidate[1])}%)" for candidate in matches
                )
                raise ValueError(
                    "Unknown model "
                    f"'{model_name}' for provider '{provider_key}'. Did you mean: {suggestion_text}?"
                )

        return model_name

    @staticmethod
    def _resolve_api_key(api_key: str | None, env_vars: Iterable[str]) -> str:
        if api_key:
            return api_key
        for env in env_vars:
            value = os.environ.get(env)
            if value:
                return value
        joined = ", ".join(env_vars)
        raise ValueError(
            "No API key provided. Pass `api_key` explicitly or set one of the following "
            f"environment variables: {joined or 'N/A'}."
        )

    @beartype
    def register_provider(
        self,
        name: str,
        *,
        api_key_env: Iterable[str] | None = None,
        base_url: str | None = None,
        extra_options: Mapping[str, Any] | None = None,
        aliases: Iterable[str] | None = None,
    ) -> None:
        """Register or override a provider configuration."""

        key = name.lower()
        self._providers[key] = ProviderConfig(
            api_key_env=tuple(api_key_env or ()),
            base_url=base_url,
            extra_options=dict(extra_options or {}),
        )

        self._aliases[key] = key
        for alias in aliases or ():
            self._aliases[alias.lower()] = key

    @beartype
    def list_providers(self, *, include_aliases: bool = False) -> tuple[str, ...]:
        """Return the configured provider identifiers.

        Args:
            include_aliases: When ``True`` the returned collection also includes
                alias names mapped to each provider. By default only canonical
                provider keys are returned.

        Returns:
            Sorted tuple of provider identifiers.
        """

        if include_aliases:
            return tuple(sorted(self._aliases.keys()))
        return tuple(sorted(self._providers.keys()))

    @beartype
    def describe_provider(self, provider: str) -> Mapping[str, Any]:
        """Return the stored metadata for ``provider``.

        Args:
            provider: Provider identifier or alias.

        Returns:
            Mapping with provider metadata (API key environment variables,
            base URL, aliases, and any default options).

        Raises:
            KeyError: If the provider is unknown.
        """

        provider_key = self._normalise_provider(provider)
        config = self._providers.get(provider_key)
        if config is None:
            raise KeyError(f"Unknown provider: {provider}")

        aliases = [
            alias
            for alias, target in self._aliases.items()
            if target == provider_key and alias != provider_key
        ]

        return {
            "provider": provider_key,
            "api_key_env": tuple(config.api_key_env),
            "base_url": config.base_url,
            "aliases": tuple(sorted(aliases)),
            "extra_options": dict(config.extra_options),
        }

    @beartype
    def list_model_names(self, provider: str) -> tuple[str, ...]:
        """Return the known model identifiers for ``provider``.

        Args:
            provider: Provider identifier or alias.

        Returns:
            Alphabetically sorted tuple of model names recognised for the
            provider. The lookup uses LiteLLM's bundled catalogue and includes
            any models exposed through provider aliases or a custom LiteLLM
            provider specified via ``extra_options['custom_llm_provider']``.
        """

        provider_key = self._normalise_provider(provider)

        potential_providers = {provider_key}

        config = self._providers.get(provider_key)
        if config and isinstance(config.extra_options, Mapping):
            custom_provider = config.extra_options.get("custom_llm_provider")
            if isinstance(custom_provider, str):
                potential_providers.add(custom_provider.lower())

        potential_providers.update(
            alias for alias, target in self._aliases.items() if target == provider_key
        )

        models: set[str] = set()
        for candidate_provider in potential_providers:
            known_models = _PROVIDER_MODEL_INDEX.get(candidate_provider)
            if known_models:
                models.update(known_models)

        return tuple(sorted(models))

    @beartype
    def create(
        self,
        provider: str,
        *,
        api_key: str | None = None,
        default_model: str | None = None,
        base_url: str | None = None,
        **default_options: Any,
    ) -> LiteLLMClient:
        """Return a client configured for ``provider`` using LiteLLM under the hood."""

        provider_key = self._normalise_provider(provider)
        config = self._providers.get(provider_key)
        if config:
            env_vars = config.api_key_env
            merged_options: Dict[str, Any] = dict(config.extra_options)
            provider_base_url = config.base_url
        else:
            env_vars = ()
            merged_options = {}
            provider_base_url = None

        resolved_api_key = self._resolve_api_key(api_key, env_vars)
        merged_options.setdefault("api_key", resolved_api_key)

        final_base_url = base_url or provider_base_url
        if final_base_url:
            merged_options.setdefault("base_url", final_base_url)

        if default_model:
            resolved_model = self._normalise_model_name(provider_key, default_model)
            merged_options.setdefault("model", resolved_model)

        merged_options.update(default_options)
        return LiteLLMClient(merged_options)
