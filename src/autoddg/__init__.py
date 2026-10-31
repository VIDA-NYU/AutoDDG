from __future__ import annotations

from .autoddg import AutoDDG
from .description import DatasetDescriptionGenerator, SearchFocusedDescription
from .evaluation import (
    GPT4oMiniPairwiseEvaluator,
    GPT4oPairwiseEvaluator,
    GPTEvaluator,
    LLaMAEvaluator,
    LLaMAPairwiseEvaluator,
    PairwiseEvaluator,
)
from .llm import ClaudeCodeClient, CodexClient, LLMClient, LocalLLMClient, OpenAICompatibleClient
from .profiling import SemanticProfiler, profile_dataset
from .topic import DatasetTopicGenerator

__version__ = "0.1.2"

__all__ = [
    "AutoDDG",
    "ClaudeCodeClient",
    "CodexClient",
    "DatasetDescriptionGenerator",
    "DatasetTopicGenerator",
    "GPTEvaluator",
    "LLaMAEvaluator",
    "PairwiseEvaluator",
    "GPT4oPairwiseEvaluator",
    "GPT4oMiniPairwiseEvaluator",
    "LLaMAPairwiseEvaluator",
    "LLMClient",
    "LocalLLMClient",
    "OpenAICompatibleClient",
    "SearchFocusedDescription",
    "SemanticProfiler",
    "profile_dataset",
    "__version__",
]
