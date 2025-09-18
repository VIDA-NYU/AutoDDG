from __future__ import annotations

from typing import Any, Dict, List, Tuple

import nltk
import numpy as np
from beartype import beartype
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from numpy.linalg import norm
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer


@beartype
class SimpleSearchEngine:
    """
    Lexical search supporting BM25 TF-IDF and cosine sparse
    """

    def __init__(self, similarity: str = "BM25") -> None:
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.qrels: List[float] = []
        self.similarity = similarity
        self.bm25_model: BM25Okapi | None = None
        self.tfidf_model = None
        self.vectorizer: TfidfVectorizer | None = None
        self.sparse_text_embedding = None
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        self.stop_words = set(stopwords.words("english"))

    def cosine_similarity(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """
        Cosine similarity for dense vectors

        Args:
            vector_a: First vector
            vector_b: Second vector

        Returns:
            Similarity score
        """
        return float(np.dot(vector_a, vector_b) / (norm(vector_a) * norm(vector_b)))

    def cosine_similarity_sparse(self, vec1: Any, vec2: Any) -> float:
        """
        Cosine similarity for sparse vectors

        Args:
            vec1: First sparse vector
            vec2: Second sparse vector

        Returns:
            Similarity score
        """

        indices1 = vec1.indices
        indices2 = vec2.indices
        values1 = vec1.values
        values2 = vec2.values

        dict1 = dict(zip(indices1, values1, strict=False))
        dict2 = dict(zip(indices2, values2, strict=False))

        common_indices = set(indices1).intersection(indices2)
        dot_product = sum(dict1[i] * dict2[i] for i in common_indices)

        norm1 = float(np.sqrt(np.sum(np.square(values1))))
        norm2 = float(np.sqrt(np.sum(np.square(values2))))

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def preprocess(self, text: str) -> List[str]:
        """
        Lowercase and tokenise text

        Args:
            text: Input string

        Returns:
            List of tokens
        """

        tokens = word_tokenize(text.lower())
        return tokens

    def index_documents(
        self,
        documents: Dict[str, Dict[str, str]],
        index_fields: Dict[str, int] | None = None,
    ) -> None:
        """
        Index documents with selected fields for retrieval

        Args:
            documents: Mapping of id -> metadata
            index_fields: Fields to index (default title and description)
        """

        if index_fields is None:
            index_fields = {"title": 1, "description": 1}

        self.documents = []
        self.doc_ids = []
        self.qrels = []

        for doc_id, doc in documents.items():
            if self.similarity == "COSINE":
                field = next((f for f, value in index_fields.items() if value == 1), None)
                embedding = doc.get(field or "", "")
                self.documents.append(embedding)
            else:
                text_parts = [
                    doc.get(field, "") for field, enabled in index_fields.items() if enabled
                ]
                preprocessed_text = " ".join(self.preprocess(" ".join(text_parts)))
                self.documents.append(preprocessed_text)

            self.doc_ids.append(doc_id)
            self.qrels.append(float(doc.get("qrel", 0)))

        if self.similarity == "BM25":
            self.bm25_model = BM25Okapi([self.preprocess(doc) for doc in self.documents])
        elif self.similarity == "TFIDF":
            self.vectorizer = TfidfVectorizer()
            self.tfidf_model = self.vectorizer.fit_transform(self.documents)

    def search(
        self, query_str: str, size: int = 10
    ) -> Tuple[List[float], List[float], List[Tuple[str, float, float]]]:
        """
        Search and return retrieved and ideal relevances and ranked docs

        Args:
            query_str: Query string or embedding
            size: Number of results to return

        Returns:
            (retrieved_relevances, ideal_relevances, ranked_docs)
        """

        if self.similarity != "COSINE":
            query_tokens = self.preprocess(query_str)
            query_str_preprocessed = " ".join(query_tokens)

        if self.similarity == "BM25" and self.bm25_model is not None:
            scores = self.bm25_model.get_scores(query_tokens)
        elif self.similarity == "TFIDF" and self.vectorizer is not None:
            query_vec = self.vectorizer.transform([query_str_preprocessed])
            scores = np.dot(self.tfidf_model, query_vec.T).toarray().flatten()
        elif self.similarity == "COSINE":
            query_embedding = query_str
            scores = [
                self.cosine_similarity_sparse(query_embedding, doc_embedding)
                for doc_embedding in self.documents
            ]
        else:
            scores = np.zeros(len(self.documents))

        ranked_docs = sorted(
            zip(self.doc_ids, scores, self.qrels, strict=False),
            key=lambda x: x[1],
            reverse=True,
        )[:size]

        retrieved_relevance = [doc[2] for doc in ranked_docs]
        ideal_relevance = sorted(self.qrels, reverse=True)[:size]

        return retrieved_relevance, ideal_relevance, ranked_docs
