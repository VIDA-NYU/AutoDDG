"""Regression tests for bugs reproduced on AutoDDG 0.1.2"""

from __future__ import annotations

from typing import Any

import pytest

from autoddg import PairwiseEvaluator
from autoddg.evaluation.base import BaseEvaluator
from autoddg.llm import LLMClient
from autoddg.profiling import SemanticProfiler
from autoddg.ranking import downstream_task_rank
from autoddg.utils import get_various_descriptions


class FakeClient(LLMClient):
    """LLMClient that returns fixed answers and records each call's temperature"""

    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply
        self.temperatures: list[float] = []

    def chat_completions_create(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.temperatures.append(temperature)
        return {
            "choices": [{"message": {"role": "assistant", "content": self.reply}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        }


class FakeOpenAI:
    """Object shaped like openai.OpenAI that always answers with `reply`"""

    def __init__(self, reply: str) -> None:
        message = type("Message", (), {"role": "assistant", "content": reply})
        choice = type("Choice", (), {"message": message})
        response = type("Response", (), {"choices": [choice], "usage": None})
        create = staticmethod(lambda **kwargs: response)
        self.chat = type("Chat", (), {"completions": type("Completions", (), {"create": create})})


# was: AttributeError: 'ClaudeCodeClient' object has no attribute 'chat'
def test_evaluators_accept_llm_clients():
    assert BaseEvaluator(FakeClient("Completeness: 9"), "m").evaluate("d") == "Completeness: 9"
    assert PairwiseEvaluator(FakeClient("B"), "m").compare("a", "b") == "B"


def test_evaluators_still_accept_openai_clients():
    assert BaseEvaluator(FakeOpenAI("Completeness: 9"), "m").evaluate("d") == "Completeness: 9"
    assert PairwiseEvaluator(FakeOpenAI("B"), "m").compare("a", "b") == "B"


# was: every answer but a bare letter counted for A, and "Both ..." counted for B
@pytest.mark.parametrize(
    ("answer", "winner"),
    [
        ("A", "A"),
        ("B", "B"),
        (" b. ", "B"),
        ("**B**", "B"),
        ("Answer: B", "B"),
        ("Description B is better.", "B"),
        ("Both are equally good.", "tie"),
        ("Neither", "tie"),
        ("A is better than B", "tie"),
        ("", "tie"),
    ],
)
def test_pairwise_answer_parsing(answer, winner):
    assert PairwiseEvaluator.parse_winner(answer) == winner


def test_ties_count_half():
    results = {"overall": [(0, 1, "tie")]}
    assert PairwiseEvaluator.compute_win_rates(results, 2) == {"overall": [0.5, 0.5]}
    assert PairwiseEvaluator.compute_elo_ratings(results, 2) == {"overall": [1500.0, 1500.0]}


# was: a single "baseline" key, overwritten by every temperature and length
def test_one_baseline_per_setting():
    client = FakeClient()
    out = get_various_descriptions(
        [0.0, 0.5], [50, 100], [True], client, "m", "a,b\n1,2", "profile", "semantic", "topic"
    )
    baselines = sorted(key for key in out if key.startswith("baseline"))
    assert baselines == [
        "baseline+temperature-0.0+num_words-100",
        "baseline+temperature-0.0+num_words-50",
        "baseline+temperature-0.5+num_words-100",
        "baseline+temperature-0.5+num_words-50",
    ]
    assert len(out) == len(client.temperatures) == 8


# was: ValueError: operands could not be broadcast together with shapes (3,) (5,)
def test_ndcg_with_fewer_documents_than_k():
    docs = ["fountain park", "taxi trip", "school"]
    result = downstream_task_rank(docs, "fountain", [2.0, 0.0, 1.0], [2, 3, 5])
    assert result[5]["ndcg"] == pytest.approx(result[3]["ndcg"])  # k > n uses all documents
    assert 0.0 < result[5]["ndcg"] <= 1.0
    assert 0.0 < result[2]["ndcg"] <= 1.0


# was: nan
def test_ndcg_without_relevant_documents():
    assert downstream_task_rank(["a", "b"], "a", [0.0, 0.0], [2]) == {2: {"ndcg": 0.0}}


# was: always 0 tokens, because the dict response was read as an object
def test_group_prompting_token_stats():
    reply = '{"x": {"Entity Type": "thing"}}'
    results, stats = SemanticProfiler(FakeClient(reply))._get_semantic_types_group([("x", ["1"])])
    assert results == {"x": {"Entity Type": "thing"}}
    assert (stats["input_tokens"], stats["output_tokens"], stats["total_tokens"]) == (100, 20, 120)


# was: the shorter prompt of a batch leaked into its answer
def test_local_batch_matches_single_calls():
    pytest.importorskip("transformers")
    from autoddg import LocalLLMClient

    try:
        client = LocalLLMClient("hf-internal-testing/tiny-random-LlamaForCausalLM", device="cpu")
    except OSError as error:  # no network or no cached model
        pytest.skip(f"tiny model unavailable: {error}")
    client.tokenizer.chat_template = None  # use the plain "User: ...\nAssistant: " format
    short = [{"role": "user", "content": "Hi"}]
    long = [{"role": "user", "content": "PROMPT_MARKER " * 40}]
    batch = client.chat_completions_create_batch("m", [short, long], max_tokens=8)
    single = client.chat_completions_create("m", short, max_tokens=8)
    answer = batch[0]["choices"][0]["message"]["content"]
    assert "User:" not in answer and "PROMPT_MARKER" not in answer
    assert answer == single["choices"][0]["message"]["content"]
    assert batch[0]["usage"]["completion_tokens"] <= 8
    assert single["usage"]["completion_tokens"] <= 8
