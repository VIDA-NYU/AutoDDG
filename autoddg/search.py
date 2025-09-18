import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from numpy.linalg import norm
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK stopwords if not already done
nltk.download("punkt")
nltk.download("stopwords")


class SimpleSearchEngine:
    def __init__(self, similarity="BM25"):
        """
        Initializes the SimpleSearchEngine.
        similarity: Choose between "BM25", "TFIDF", "COSINE".
        """
        self.documents = []  # List of documents
        self.doc_ids = []  # Track document IDs
        self.qrels = []  # Track query relevance
        self.similarity = similarity
        self.bm25_model = None
        self.tfidf_model = None
        self.vectorizer = None
        self.sparse_text_embedding = None
        self.stop_words = set(stopwords.words("english"))  # NLTK stop words

    def cosine_similarity(self, a, b):
        """Computes the cosine similarity between vectors a and b."""
        return np.dot(a, b) / (norm(a) * norm(b))

    def cosine_similarity_sparse(self, vec1, vec2):
        # vec1 and vec2 are dictionaries with 'values' and 'indices'
        # Step 1: Compute dot product (only over common indices)
        indices1 = vec1.indices
        indices2 = vec2.indices
        values1 = vec1.values
        values2 = vec2.values

        # Create dictionaries for quick lookup
        dict1 = dict(zip(indices1, values1, strict=False))
        dict2 = dict(zip(indices2, values2, strict=False))

        # Compute dot product
        common_indices = set(indices1).intersection(indices2)
        dot_product = sum(dict1[i] * dict2[i] for i in common_indices)

        # Step 2: Compute norms
        norm1 = np.sqrt(np.sum(np.square(values1)))
        norm2 = np.sqrt(np.sum(np.square(values2)))

        # Step 3: Calculate cosine similarity
        if norm1 == 0 or norm2 == 0:  # Avoid division by zero
            return 0.0
        return dot_product / (norm1 * norm2)

    def preprocess(self, text):
        """Tokenizes and removes stop words from the text."""
        tokens = word_tokenize(text.lower())  # Lowercase and tokenize
        # tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]  # Remove stop words
        # tokens = [word for word in tokens if word not in self.stop_words]  # Remove stop words
        return tokens

    def index_documents(self, documents, index_fields=None):
        """
        Indexes documents.
        documents: List of dictionaries containing 'doc_id', 'title', and 'description'.
        """
        if index_fields is None:
            index_fields = {"title": 1, "description": 1}

        self.documents = []  # Reset documents
        self.doc_ids = []  # Reset document IDs
        for doc_id in documents:
            doc = documents[doc_id]

            if self.similarity == "COSINE":
                field = [f for f in index_fields if index_fields[f] == 1][0]
                # print(f"field: {field}")
                embedding = doc.get(field)
                self.documents.append(embedding)
            else:
                text = ""
                for field in index_fields:
                    if index_fields[field] == 1:
                        text += doc.get(field) + " "
                preprocessed_text = " ".join(self.preprocess(text))
                self.documents.append(preprocessed_text)

            self.doc_ids.append(doc_id)
            self.qrels.append(doc["qrel"])

        # Build models based on the chosen similarity
        if self.similarity == "BM25":
            self.bm25_model = BM25Okapi(
                [self.preprocess(doc) for doc in self.documents]
            )
        elif self.similarity == "TFIDF":
            self.vectorizer = TfidfVectorizer()
            self.tfidf_model = self.vectorizer.fit_transform(self.documents)
        # elif self.similarity == "COSINE":
        #     self.sparse_text_embedding = SparseTextEmbedding(model_name="prithvida/Splade_PP_en_v1")
        #     self.documents = list(self.sparse_text_embedding.embed(self.documents))

    def search(self, query_str, size=10):
        """
        Searches the indexed documents using weighted fields.
        query_str: Search query string.
        field_weights: Dictionary of field weights (e.g., title=3, description=1).
        size: Number of results to return.
        """
        # Preprocess the query
        if self.similarity != "COSINE":
            query = self.preprocess(query_str)
            query_str_preprocessed = " ".join(query)

        if self.similarity == "BM25":
            # Perform BM25 search
            scores = self.bm25_model.get_scores(query)
        elif self.similarity == "TFIDF":
            # Perform TF-IDF search
            query_vec = self.vectorizer.transform([query_str_preprocessed])
            scores = np.dot(self.tfidf_model, query_vec.T).toarray().flatten()
        elif self.similarity == "COSINE":
            # Perform cosine similarity search
            # query_embedding = list(self.sparse_text_embedding.embed(query_str_preprocessed))[0]
            query_embedding = query_str
            # print(f"query_embedding: {query_embedding}")
            # print(f"doc_embedding: {self.documents[0]}")
            scores = [
                self.cosine_similarity_sparse(query_embedding, doc_embedding)
                for doc_embedding in self.documents
            ]
        # Sort documents by score and return top results
        ranked_docs = sorted(
            zip(self.doc_ids, scores, self.qrels, strict=False),
            key=lambda x: x[1],
            reverse=True,
        )[:size]
        retrieved_relevance = [doc[2] for doc in ranked_docs]
        ideal_relevance = sorted(self.qrels, reverse=True)[:size]

        return retrieved_relevance, ideal_relevance, ranked_docs
