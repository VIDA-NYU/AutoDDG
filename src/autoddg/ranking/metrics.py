from __future__ import annotations

import math
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from beartype import beartype
from rank_bm25 import BM25Okapi


@beartype
def compute_dcg(relevances: Iterable[float], p: int) -> float:
    """
    Discounted cumulative gain at rank p

    Args:
        relevances: Relevance scores
        p: Cut-off rank

    Returns:
        DCG value
    """

    dcg = 0.0
    for index, relevance in enumerate(relevances):
        if index >= p:
            break
        dcg += (2**relevance - 1) / math.log2(index + 2)
    return dcg


@beartype
def compute_avg_single_Q(
    stats: Dict[str, Dict[str, Dict[str, List[float]]]],
    description_version_key: str,
    Q_key: str,
) -> pd.DataFrame:
    """
    Average nested metrics into a DataFrame by index version

    Args:
        stats: Nested metrics dict
        description_version_key: Key selecting description version
        Q_key: Key selecting metric group

    Returns:
        DataFrame of averaged metrics
    """

    averages: Dict[str, Dict[str, float]] = {}
    ndcg_dicts = stats[description_version_key][Q_key]
    for index_version, ndcg_metric in ndcg_dicts.items():
        averages[index_version] = {
            key: float(np.average(scores)) for key, scores in ndcg_metric.items()
        }
    return pd.DataFrame(averages)


@beartype
def compute_ndcg(
    retrieved_relevances: Iterable[float], ideal_relevances: Iterable[float], p: int
) -> float:
    """
    Normalised DCG at rank p

    Args:
        retrieved_relevances: Relevances by retrieved order
        ideal_relevances: Relevances by ideal order
        p: Cut-off rank

    Returns:
        nDCG value
    """

    dcg = compute_dcg(retrieved_relevances, p)
    idcg = compute_dcg(ideal_relevances, p)
    return dcg / idcg if idcg > 0 else 0.0


@beartype
def downstream_task_rank(
    documents: List[str],
    query: str,
    relevances: List[float],
    ks: Iterable[int],
    debug: bool = False,
) -> Dict[int, Dict[str, float]]:
    """
    BM25 ranking with nDCG@k over the inputted documents

    Args:
        documents: List of document texts
        query: Query string
        relevances: Ground-truth relevance scores
        ks: List of cut-off values
        debug: Print debug output if True

    Returns:
        Mapping k -> metrics
    """

    def _compute_ndcg(relevance_true: List[float], relevance_test: List[float], k: int) -> float:
        ideal_dcg = np.sum(np.array(relevance_true) / np.log2(np.arange(2, k + 2)))
        dcg = np.sum(np.array(relevance_test) / np.log2(np.arange(2, k + 2)))
        return float(dcg / ideal_dcg)

    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    if debug:
        print(tokenized_query)

    scores = bm25.get_scores(tokenized_query)
    sorted_indices = np.argsort(scores)[::-1]
    sorted_rel_true = sorted(relevances, reverse=True)
    sorted_rel_test = np.array(relevances)[sorted_indices].tolist()

    results: Dict[int, Dict[str, float]] = {}
    for k in ks:
        ndcg = _compute_ndcg(sorted_rel_true[:k], sorted_rel_test[:k], k)
        results[k] = {"ndcg": ndcg}
    return results
