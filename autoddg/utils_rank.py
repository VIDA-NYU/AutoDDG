import math
from collections import defaultdict

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi


def extract_query_rel(qrel_file_path):
    query_rel = defaultdict(lambda: defaultdict(list))
    query_rel_list = []
    with open(qrel_file_path) as f:
        lines = f.readlines()[1:]
        for line in lines:
            t, r, d, _ = line.split(",")
            d = "/".join(d.split("/")[1:2])
            r = float(r)
            query_rel[t][d].append(r)
            query_rel_list.append((d, t, r))
    return query_rel, query_rel_list


def compute_dcg(relevances, p):
    dcg = 0.0
    for i in range(min(p, len(relevances))):
        dcg += (2 ** relevances[i] - 1) / math.log2(
            i + 2
        )  # i+2 because log_2(rank + 1)
    return dcg


def compute_avg_single_Q(stats, description_version_key, Q_key):
    averages = {}
    ndcg_dicts = stats[description_version_key][Q_key]
    for index_version, ndcg_metric in ndcg_dicts.items():
        averages[index_version] = {
            k: np.average(scores) for k, scores in ndcg_metric.items()
        }
    averages = pd.DataFrame(averages)
    return averages


def compute_ndcg(retrieved_relevances, ideal_relevances, p):
    dcg = compute_dcg(retrieved_relevances, p)
    idcg = compute_dcg(ideal_relevances, p)
    return dcg / idcg if idcg > 0 else 0.0


def downstream_task_rank(documents, query, relevances, ks, debug=False):
    def _compute_ndcg(relevantce_true, relevantce_test, k):
        ideal_dcg = np.sum(relevantce_true / np.log2(np.arange(2, k + 2)))
        dcg = np.sum(relevantce_test / np.log2(np.arange(2, k + 2)))
        ndcg = dcg / ideal_dcg
        return ndcg

    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    if debug:
        print(tokenized_query)

    scores = bm25.get_scores(tokenized_query)
    sorted_indices = np.argsort(scores)[::-1]
    sorted_rel_true = sorted(relevances, reverse=True)
    sorted_rel_test = np.array(relevances)[sorted_indices].tolist()

    results = {}
    for k in ks:
        # y_true = [1 if i != 0 else 0 for i in sorted_rel_true[:k]]
        # y_pred = [1 if i != 0 else 0 for i in sorted_rel_test[:k]]
        # precision = precision_score(y_true, y_pred)
        # recall = recall_score(y_true, y_pred)
        ndcg = _compute_ndcg(sorted_rel_true[:k], sorted_rel_test[:k], k)
        results[k] = {"ndcg": ndcg}
        # results[k] = {"precision": precision,
        #               "recall": recall,
        #               "ndcg": ndcg}
        # if debug:
        #     print("======k@", k)
        #     print("y_true", y_true)
        #     print("y_pred", y_pred)
        #     print("ndcg", ndcg)
    return results
