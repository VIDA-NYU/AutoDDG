from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from beartype import beartype


@beartype
def extract_query_rel(
    qrel_file_path: str,
) -> Tuple[Dict[str, Dict[str, List[float]]], List[Tuple[str, str, float]]]:
    """
    Parse a qrel CSV into nested and flat relevance structures

    Args:
        qrel_file_path: Path to CSV file

    Returns:
        (nested: topic -> document -> [scores], flat: [(document, topic, score)])
    """

    query_rel: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    query_rel_list: List[Tuple[str, str, float]] = []
    with open(qrel_file_path, encoding="utf-8") as file:
        lines = file.readlines()[1:]
        for line in lines:
            topic, relevance, document, _ = line.split(",")
            document = "/".join(document.split("/")[1:2])
            score = float(relevance)
            query_rel[topic][document].append(score)
            query_rel_list.append((document, topic, score))
    return query_rel, query_rel_list
