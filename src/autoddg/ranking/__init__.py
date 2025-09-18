from __future__ import annotations

from qrels import extract_query_rel

from .metrics import (
    compute_avg_single_Q,
    compute_dcg,
    compute_ndcg,
    downstream_task_rank,
)

__all__ = [
    "compute_avg_single_Q",
    "compute_dcg",
    "compute_ndcg",
    "downstream_task_rank",
    "extract_query_rel",
]
