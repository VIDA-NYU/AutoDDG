from __future__ import annotations

import logging
import random
import re
from itertools import combinations
from typing import Any, Literal

from beartype import beartype

from ..utils import load_prompts
from .base import BaseEvaluator

logger = logging.getLogger(__name__)


def _score(winner: str) -> float:
    """Score of the first description of a pair: 1 for "A", 0 for "B", 0.5 for a tie"""
    return {"A": 1.0, "B": 0.0}.get(winner, 0.5)


@beartype
class PairwiseEvaluator(BaseEvaluator):
    """
    Pairwise evaluator for comparing multiple descriptions

    Compares descriptions in pairs and computes win rates and ELO ratings.
    Supports different sampling strategies for pair selection.

    Args:
        client: An ``LLMClient`` or an OpenAI-compatible client instance
        model_name: Model name to use for evaluation
    """

    def __init__(self, client: Any, model_name: str = "gpt-4o") -> None:
        super().__init__(client=client, model_name=model_name)
        # Load pairwise-specific prompts from prompts.yaml
        prompts = load_prompts()
        pairwise_prompts = prompts["pairwise_evaluation"]
        self._system_message = pairwise_prompts["system_message"].strip()
        self._pairwise_template = pairwise_prompts["user_prompt"]

    def compare(self, desc_a: str, desc_b: str) -> Literal["A", "B", "tie"]:
        """
        Compare two descriptions and return which is better

        Args:
            desc_a: First description
            desc_b: Second description

        Returns:
            "A" or "B" for the better description, or "tie" when the answer names neither
            or both (it is then counted as half a win for each)
        """
        user_content = self._pairwise_template.format(desc_a=desc_a, desc_b=desc_b)
        answer = self._chat(user_content, temperature=0.0)
        winner = self.parse_winner(answer)
        if winner == "tie":
            logger.warning("Pairwise judge gave no clear winner, counted as a tie: %r", answer)
        return winner

    @staticmethod
    def parse_winner(answer: str) -> Literal["A", "B", "tie"]:
        """
        Read the winner from a judge's answer

        Accepts a bare letter with optional formatting ("B", "**B**", "B.") or an answer that
        names exactly one of the capital letters A and B as a word ("Answer: B",
        "Description B is better"). Anything else, including answers naming both, is a tie.

        Args:
            answer: Raw model answer

        Returns:
            "A", "B" or "tie"
        """
        stripped = re.sub(r"[^A-Za-z]", "", answer).upper()
        if stripped in ("A", "B"):
            return stripped
        # Capital letters only: a lowercase "a" is usually the article
        named = set(re.findall(r"\b([AB])\b", answer))
        if len(named) == 1:
            return named.pop()
        return "tie"

    def compare_batch(
        self,
        descriptions: list[str],
        strategy: Literal["full", "no_repeat", "symmetric"] = "no_repeat",
        max_pairs: int | None = None,
    ) -> dict[str, list[tuple[int, int, str]]]:
        """
        Compare multiple descriptions in pairs

        Args:
            descriptions: List of descriptions to compare
            strategy: Pairing strategy:
                - "full": All possible pairs
                - "no_repeat": Random sample of pairs (default)
                - "symmetric": Random sample with symmetric pairs (i, j) and (j, i)
            max_pairs: Maximum number of pairs to compare (for sampling strategies)

        Returns:
            Dictionary with results as list of (i, j, winner) tuples
            where winner is "A" (description i), "B" (description j) or "tie"
        """
        pairs = self.sample_pairs(len(descriptions), strategy=strategy, max_pairs=max_pairs)
        results: list[tuple[int, int, str]] = []

        for i, j in pairs:
            results.append((i, j, self.compare(descriptions[i], descriptions[j])))

        return {"overall": results}

    @staticmethod
    def sample_pairs(
        n: int,
        strategy: Literal["full", "no_repeat", "symmetric"] = "no_repeat",
        max_pairs: int | None = None,
    ) -> list[tuple[int, int]]:
        """
        Sample pairs of indices for comparison

        Args:
            n: Number of descriptions
            strategy: Pairing strategy
            max_pairs: Maximum number of pairs (for sampling strategies)

        Returns:
            List of (i, j) tuples representing pairs to compare
        """
        if strategy == "full":
            pairs = list(combinations(range(n), 2))
        elif strategy == "no_repeat":
            all_pairs = list(combinations(range(n), 2))
            random.shuffle(all_pairs)
            pairs = all_pairs[: max_pairs or len(all_pairs)]
        elif strategy == "symmetric":
            all_pairs = list(combinations(range(n), 2))
            random.shuffle(all_pairs)
            selected = all_pairs[: (max_pairs // 2) if max_pairs else len(all_pairs)]
            pairs = selected + [(j, i) for i, j in selected]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return pairs

    @staticmethod
    def compute_win_rates(
        results: dict[str, list[tuple[int, int, str]]], n_desc: int
    ) -> dict[str, list[float]]:
        """
        Compute win rates for each description (a tie counts as half a win)

        Args:
            results: Dictionary mapping aspect to list of (i, j, winner) tuples
            n_desc: Number of descriptions

        Returns:
            Dictionary mapping aspect to list of win rates (one per description)
        """
        win_rates: dict[str, list[float]] = {}
        for aspect, res in results.items():
            wins = [0.0] * n_desc
            plays = [0] * n_desc
            for i, j, winner in res:
                s_i = _score(winner)
                wins[i] += s_i
                wins[j] += 1.0 - s_i
                plays[i] += 1
                plays[j] += 1
            win_rates[aspect] = [wins[k] / plays[k] if plays[k] else 0.0 for k in range(n_desc)]
        return win_rates

    @staticmethod
    def compute_elo_ratings(
        results: dict[str, list[tuple[int, int, str]]],
        n_desc: int,
        k: int = 32,
        init_rating: int = 1500,
    ) -> dict[str, list[float]]:
        """
        Compute ELO ratings for each description

        Args:
            results: Dictionary mapping aspect to list of (i, j, winner) tuples
            n_desc: Number of descriptions
            k: ELO K-factor (default: 32)
            init_rating: Initial ELO rating (default: 1500)

        Returns:
            Dictionary mapping aspect to list of ELO ratings (one per description)
        """

        def expect(r1: float, r2: float) -> float:
            return 1 / (1 + 10 ** ((r2 - r1) / 400))

        elo_scores: dict[str, list[float]] = {}
        for aspect, res in results.items():
            ratings = [float(init_rating)] * n_desc
            for i, j, winner in res:
                r_i, r_j = ratings[i], ratings[j]
                e_i = expect(r_i, r_j)
                s_i = _score(winner)
                s_j = 1.0 - s_i
                ratings[i] += k * (s_i - e_i)
                ratings[j] += k * (s_j - (1 - e_i))
            elo_scores[aspect] = ratings
        return elo_scores


@beartype
class GPT4oPairwiseEvaluator(PairwiseEvaluator):
    """
    Pairwise evaluator using GPT-4o model
    """

    def __init__(self, gpt4_api_key: str, model_name: str = "gpt-4o") -> None:
        import openai

        client = openai.OpenAI(api_key=gpt4_api_key)
        super().__init__(client=client, model_name=model_name)


@beartype
class GPT4oMiniPairwiseEvaluator(PairwiseEvaluator):
    """
    Pairwise evaluator using GPT-4o-mini model (faster and cheaper)
    """

    def __init__(self, gpt4_api_key: str, model_name: str = "gpt-4o-mini") -> None:
        import openai

        client = openai.OpenAI(api_key=gpt4_api_key)
        super().__init__(client=client, model_name=model_name)


@beartype
class LLaMAPairwiseEvaluator(PairwiseEvaluator):
    """
    Pairwise evaluator using DeepInfra LLaMA
    """

    def __init__(
        self,
        llama_api_key: str,
        model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct",
    ) -> None:
        import openai

        client = openai.OpenAI(
            api_key=llama_api_key, base_url="https://api.deepinfra.com/v1/openai"
        )
        super().__init__(client=client, model_name=model_name)
