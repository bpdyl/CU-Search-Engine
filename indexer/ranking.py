"""
Ranking Algorithms for Document Retrieval

Implements TF-IDF and BM25 ranking algorithms for scoring documents.
"""
import math
from collections import defaultdict

import config


class TFIDFRanker:
    """
    TF-IDF based document ranker with field weighting support.
    """

    def __init__(self, inverted_index):
        """
        Initialize the ranker.

        Args:
            inverted_index: InvertedIndex instance
        """
        self.index = inverted_index
        self.field_weights = inverted_index.field_weights

    def calculate_tf(self, term_freq, doc_length):
        """
        Calculate Term Frequency (normalized).

        Args:
            term_freq: Raw term frequency
            doc_length: Document length

        Returns:
            Normalized TF value
        """
        if doc_length == 0:
            return 0
        # Log normalization
        return 1 + math.log(term_freq) if term_freq > 0 else 0

    def calculate_idf(self, term):
        """
        Calculate Inverse Document Frequency.

        Args:
            term: Search term

        Returns:
            IDF value
        """
        return self.index.get_idf(term)

    def score_document(self, doc_id, query_terms, query_term_freqs):
        """
        Calculate TF-IDF score for a document given query terms.

        Args:
            doc_id: Document identifier
            query_terms: List of query terms
            query_term_freqs: Dict of term -> frequency in query

        Returns:
            Document score
        """
        score = 0.0
        doc_length = self.index.doc_lengths.get(doc_id, 1)

        for term in query_terms:
            # Get postings for term
            postings = self.index.get_postings(term)

            # Find this document in postings
            for did, freq, field in postings:
                if did == doc_id:
                    # Calculate TF-IDF
                    tf = self.calculate_tf(freq, doc_length)
                    idf = self.calculate_idf(term)
                    query_tf = query_term_freqs.get(term, 1)

                    score += tf * idf * query_tf

                    # Boost for title matches
                    if 'title' in field:
                        score *= 1.5

                    break

        return score


class BM25Ranker:
    """
    BM25 (Best Matching 25) ranking algorithm.

    More sophisticated ranking than TF-IDF, with document length normalization.
    """

    def __init__(self, inverted_index, k1=1.5, b=0.75):
        """
        Initialize BM25 ranker.

        Args:
            inverted_index: InvertedIndex instance
            k1: Term frequency saturation parameter
            b: Document length normalization parameter
        """
        self.index = inverted_index
        self.k1 = k1
        self.b = b

    def calculate_idf(self, term):
        """
        Calculate IDF using BM25 formula.

        Args:
            term: Search term

        Returns:
            IDF value
        """
        N = self.index.total_docs
        df = self.index.get_document_frequency(term)

        if df == 0:
            return 0

        # BM25 IDF formula
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def score_document(self, doc_id, query_terms, query_term_freqs=None):
        """
        Calculate BM25 score for a document.

        Args:
            doc_id: Document identifier
            query_terms: List of query terms
            query_term_freqs: Optional dict of term frequencies in query

        Returns:
            BM25 score
        """
        score = 0.0
        doc_length = self.index.doc_lengths.get(doc_id, 0)
        avg_dl = self.index.avg_doc_length

        if avg_dl == 0:
            return 0

        for term in query_terms:
            # Get IDF
            idf = self.calculate_idf(term)

            if idf == 0:
                continue

            # Get term frequency in document
            tf = 0
            field_match = ''
            postings = self.index.get_postings(term)

            for did, freq, field in postings:
                if did == doc_id:
                    tf = freq
                    field_match = field
                    break

            if tf == 0:
                continue

            # BM25 score calculation
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / avg_dl))

            term_score = idf * (numerator / denominator)

            # Field boosting
            if 'title' in field_match:
                term_score *= 2.0
            elif 'authors' in field_match:
                term_score *= 1.5

            score += term_score

        return score


class HybridRanker:
    """
    Hybrid ranker combining TF-IDF and BM25 scores.
    """

    def __init__(self, inverted_index, tfidf_weight=0.4, bm25_weight=0.6):
        """
        Initialize hybrid ranker.

        Args:
            inverted_index: InvertedIndex instance
            tfidf_weight: Weight for TF-IDF score (0-1)
            bm25_weight: Weight for BM25 score (0-1)
        """
        self.tfidf = TFIDFRanker(inverted_index)
        self.bm25 = BM25Ranker(inverted_index)
        self.tfidf_weight = tfidf_weight
        self.bm25_weight = bm25_weight

    def score_document(self, doc_id, query_terms, query_term_freqs):
        """
        Calculate hybrid score combining TF-IDF and BM25.

        Args:
            doc_id: Document identifier
            query_terms: List of query terms
            query_term_freqs: Dict of term frequencies in query

        Returns:
            Combined score
        """
        tfidf_score = self.tfidf.score_document(doc_id, query_terms, query_term_freqs)
        bm25_score = self.bm25.score_document(doc_id, query_terms, query_term_freqs)

        return (self.tfidf_weight * tfidf_score) + (self.bm25_weight * bm25_score)


def get_ranker(inverted_index, algorithm='bm25'):
    """
    Factory function to get appropriate ranker.

    Args:
        inverted_index: InvertedIndex instance
        algorithm: Ranking algorithm ('tfidf', 'bm25', or 'hybrid')

    Returns:
        Ranker instance
    """
    if algorithm == 'tfidf':
        return TFIDFRanker(inverted_index)
    elif algorithm == 'bm25':
        return BM25Ranker(inverted_index)
    elif algorithm == 'hybrid':
        return HybridRanker(inverted_index)
    else:
        return BM25Ranker(inverted_index)  # Default to BM25
