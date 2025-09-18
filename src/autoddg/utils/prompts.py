from __future__ import annotations

from importlib import resources
from typing import Any, Dict

import yaml
from beartype import beartype

_PROMPT_CACHE: Dict[str, Any] | None = None


@beartype
def load_prompts() -> Dict[str, Any]:
    """
    Load and cache prompts.yaml from the package resources

    Returns:
        Prompt configuration
    """

    global _PROMPT_CACHE
    if _PROMPT_CACHE is None:
        with (
            resources.files("autoddg.configurations")
            .joinpath("prompts.yaml")
            .open("r", encoding="utf-8") as stream
        ):
            _PROMPT_CACHE = yaml.safe_load(stream)
    return _PROMPT_CACHE
