from __future__ import annotations

from typing import Any, Dict, List

from beartype import beartype

from .base import LLMClient


@beartype
class OpenAICompatibleClient(LLMClient):
    """Wrapper for OpenAI-compatible API clients"""

    def __init__(self, client: Any) -> None:
        """
        Initialize with an OpenAI-compatible client

        Args:
            client: OpenAI-compatible client instance (e.g., openai.OpenAI)
        """
        self.client = client

    def chat_completions_create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate a chat completion using OpenAI-compatible API

        Args:
            model: Model identifier
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature
            **kwargs: Additional parameters passed to the API client

        Returns:
            Response dict compatible with OpenAI API format
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )

        # Convert response to dict format
        return {
            "choices": [
                {
                    "message": {
                        "role": response.choices[0].message.role,
                        "content": response.choices[0].message.content,
                    }
                }
            ],
            "usage": (
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                if hasattr(response, "usage") and response.usage
                else None
            ),
        }

