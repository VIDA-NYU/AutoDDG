from __future__ import annotations

from .autoddg import AutoDDG
from .description import DatasetDescription, DatasetDescriptionGenerator, SearchFocusedDescription
from .evaluation import (
    GPT4oMiniPairwiseEvaluator,
    GPT4oPairwiseEvaluator,
    GPTEvaluator,
    LLaMAEvaluator,
    LLaMAPairwiseEvaluator,
    PairwiseEvaluator,
    check_description,
    check_numbers,
)
from .llm import ClaudeCodeClient, CodexClient, LLMClient, LocalLLMClient, OpenAICompatibleClient
from .profiling import SemanticProfiler, profile_dataset
from .report import DescriptionReport
from .topic import DatasetTopicGenerator

__version__ = "0.1.2"

__all__ = [
    "AutoDDG",
    "ClaudeCodeClient",
    "CodexClient",
    "DatasetDescription",
    "DatasetDescriptionGenerator",
    "DescriptionReport",
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
    "check_description",
    "check_numbers",
    "__version__",
]
