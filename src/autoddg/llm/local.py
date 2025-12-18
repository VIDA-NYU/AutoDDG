from __future__ import annotations

import warnings
from typing import Any

from beartype import beartype

from .base import LLMClient

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn(
        "transformers and torch not available. " "Install with: pip install transformers torch",
        stacklevel=2,
    )


@beartype
class LocalLLMClient(LLMClient):
    """Local LLM client using transformers library"""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        torch_dtype: str | None = None,
        trust_remote_code: bool = False,
        **model_kwargs: Any,
    ) -> None:
        """
        Initialize local LLM client

        Args:
            model_name: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-7B-Instruct")
            device: Device to run on ("cuda", "cpu", or None for auto)
            torch_dtype: Data type ("float16", "bfloat16", "float32", or None for auto)
            trust_remote_code: Whether to trust remote code in model
            **model_kwargs: Additional arguments for model loading
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for local LLM support. "
                "Install with: pip install transformers torch"
            )

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Determine dtype
        if torch_dtype is None:
            if self.device == "cuda":
                self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                self.dtype = torch.float32
        else:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            self.dtype = dtype_map.get(torch_dtype.lower(), torch.float32)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()

    def _format_messages(self, messages: list[dict[str, str]]) -> tuple[str, int]:
        """
        Format messages into a prompt string using the tokenizer's chat template
        if available

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            Tuple of (formatted prompt string, prompt token count)
        """
        # Try to use tokenizer's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            # Format messages for chat template
            formatted_messages = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                # Map OpenAI roles to standard roles
                if role == "system":
                    formatted_messages.append({"role": "system", "content": content})
                elif role == "user":
                    formatted_messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    formatted_messages.append({"role": "assistant", "content": content})

            prompt = self.tokenizer.apply_chat_template(
                formatted_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
            return prompt, prompt_tokens
        else:
            # Fallback to simple formatting
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role == "system":
                    prompt_parts.append(f"System: {content}\n")
                elif role == "user":
                    prompt_parts.append(f"User: {content}\n")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}\n")

            prompt_parts.append("Assistant: ")
            prompt = "".join(prompt_parts)
            prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
            return prompt, prompt_tokens

    def chat_completions_create(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate a chat completion using local model

        Args:
            model: Model identifier (ignored, uses self.model_name)
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Response dict compatible with OpenAI API format
        """
        # Format messages into prompt
        prompt, prompt_tokens = self._format_messages(messages)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=(
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id
                ),
                **kwargs,
            )

        # Decode only the new tokens (response)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Calculate token usage
        total_tokens = outputs.shape[1]
        completion_tokens = total_tokens - prompt_tokens

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    }
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

    def chat_completions_create_batch(
        self,
        model: str,
        messages_list: list[list[dict[str, str]]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Generate chat completions for multiple prompts in batch using local model

        Args:
            model: Model identifier (ignored, uses self.model_name)
            messages_list: List of message lists, each containing message dicts
                with 'role' and 'content' keys
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            List of response dicts compatible with OpenAI API format
        """
        # Format all prompts
        prompts = []
        prompt_token_counts = []
        for messages in messages_list:
            prompt, prompt_tokens = self._format_messages(messages)
            prompts.append(prompt)
            prompt_token_counts.append(prompt_tokens)

        # Set up tokenizer for batch processing
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        try:
            # Tokenize all prompts
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
                self.device
            )

            # Generate in batch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **kwargs,
                )

            # Decode responses
            responses = []
            for i, (_prompt, prompt_tokens) in enumerate(
                zip(prompts, prompt_token_counts, strict=True)
            ):
                input_length = inputs["attention_mask"][i].sum().item()
                generated_tokens = outputs[i][input_length:]
                response_text = self.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                ).strip()

                # Calculate token usage
                # Note: outputs[i] includes both input and generated tokens
                # input_length is the actual input length (excluding padding)
                completion_tokens = len(generated_tokens)
                total_tokens = prompt_tokens + completion_tokens

                responses.append(
                    {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": response_text,
                                }
                            }
                        ],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                        },
                    }
                )

            return responses
        finally:
            # Restore original padding side
            self.tokenizer.padding_side = original_padding_side
