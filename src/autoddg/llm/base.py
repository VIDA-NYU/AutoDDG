from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from beartype import beartype


@beartype
class LLMClient(ABC):
    """Abstract base class for LLM clients"""

    @abstractmethod
    def chat_completions_create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate a chat completion

        Args:
            model: Model identifier
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature
            **kwargs: Additional model-specific parameters

        Returns:
            Response dict with 'choices' key containing list of message dicts
        """
        pass

