"""
Query Processor for Search Engine

Handles query parsing, preprocessing, and retrieval with ranking.
Supports partial matching and synonym expansion.
"""
from collections import defaultdict

from indexer.preprocessor import TextPreprocessor
from indexer.ranking import get_ranker, BM25Ranker
import config


class QueryProcessor:
    """
    Processes search queries and retrieves ranked results.
    """

    def __init__(self, inverted_index, ranking_algorithm='bm25'):
        """
        Initialize the query processor.

        Args:
            inverted_index: InvertedIndex instance
            ranking_algorithm: Ranking algorithm to use ('tfidf', 'bm25', 'hybrid')
        """
        self.index = inverted_index
        self.ranker = get_ranker(inverted_index, ranking_algorithm)
        self.preprocessor = TextPreprocessor(
            use_stemming=True,
            use_lemmatization=True,
            expand_synonyms=True
        )

    def parse_query(self, query):
        """
        Parse and preprocess query string.

        Args:
            query: Raw query string

        Returns:
            Tuple of (processed_tokens, original_tokens)
        """
        # Clean and tokenize
        original = self.preprocessor.tokenize(
            self.preprocessor.clean_text(query)
        )

        # Full preprocessing for matching
        processed = self.preprocessor.preprocess_for_query(query)

        return processed, original

    def search(self, query, limit=None):
        """
        Search for documents matching the query.

        Args:
            query: Search query string
            limit: Maximum number of results (default from config)

        Returns:
            List of (doc_id, document_data, score) tuples
        """
        if limit is None:
            limit = config.SEARCH_RESULTS_LIMIT

        if not query or not query.strip():
            return []

        # Parse query
        query_terms, original_terms = self.parse_query(query)

        if not query_terms:
            return []

        # Calculate term frequencies in query
        query_term_freqs = defaultdict(int)
        for term in query_terms:
            query_term_freqs[term] += 1

        # Find candidate documents
        candidate_docs = set()
        for term in query_terms:
            postings = self.index.get_postings(term)
            for doc_id, _, _ in postings:
                candidate_docs.add(doc_id)

        # Also search for partial matches
        all_terms = self.index.get_all_terms()
        for term in query_terms:
            partial_matches = self.preprocessor.get_partial_matches(term, all_terms)
            for match in partial_matches:
                postings = self.index.index.get(match, [])
                for doc_id, _, _ in postings:
                    candidate_docs.add(doc_id)

        if not candidate_docs:
            return []

        # Score documents
        results = []
        for doc_id in candidate_docs:
            doc = self.index.get_document(doc_id)
            if doc:
                score = self.ranker.score_document(doc_id, query_terms, query_term_freqs)

                # Bonus for multi-term matches
                matched_terms = 0
                for term in query_terms:
                    postings = self.index.get_postings(term)
                    for did, _, _ in postings:
                        if did == doc_id:
                            matched_terms += 1
                            break

                # Boost score based on term coverage
                if len(query_terms) > 1:
                    coverage = matched_terms / len(query_terms)
                    score *= (1 + coverage)

                results.append((doc_id, doc, score))

        # Sort by score descending
        results.sort(key=lambda x: x[2], reverse=True)

        return results[:limit]

    def search_by_field(self, query, field):
        """
        Search within a specific field.

        Args:
            query: Search query string
            field: Field to search ('title', 'authors', 'year', etc.)

        Returns:
            List of matching documents
        """
        query_terms, _ = self.parse_query(query)

        if not query_terms:
            return []

        results = []

        for term in query_terms:
            postings = self.index.get_postings(term)
            for doc_id, freq, posting_field in postings:
                if field in posting_field:
                    doc = self.index.get_document(doc_id)
                    if doc:
                        results.append((doc_id, doc, freq))

        # Deduplicate and sort
        seen = set()
        unique_results = []
        for doc_id, doc, freq in results:
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append((doc_id, doc, freq))

        unique_results.sort(key=lambda x: x[2], reverse=True)
        return unique_results

    def search_by_author(self, author_name):
        """
        Search for publications by a specific author.

        Args:
            author_name: Author name to search for

        Returns:
            List of publications by the author
        """
        return self.search_by_field(author_name, 'authors')

    def search_by_year(self, year):
        """
        Search for publications from a specific year.

        Args:
            year: Publication year

        Returns:
            List of publications from that year
        """
        results = []
        year_str = str(year)

        for doc_id, doc in self.index.documents.items():
            if str(doc.get('year', '')) == year_str:
                results.append((doc_id, doc, 1.0))

        return results

    def get_suggestions(self, partial_query, limit=5):
        """
        Get query suggestions based on partial input.

        Args:
            partial_query: Partial query string
            limit: Maximum suggestions to return

        Returns:
            List of suggested terms
        """
        if not partial_query:
            return []

        query_lower = partial_query.lower().strip()
        all_terms = self.index.get_all_terms()

        suggestions = []
        for term in all_terms:
            if term.startswith(query_lower):
                suggestions.append(term)
            elif query_lower in term:
                suggestions.append(term)

        # Sort by length (shorter = more relevant)
        suggestions.sort(key=len)

        return suggestions[:limit]

    def highlight_matches(self, text, query_terms):
        """
        Highlight matching terms in text.

        Args:
            text: Original text
            query_terms: List of query terms to highlight

        Returns:
            Text with HTML highlighting
        """
        if not text or not query_terms:
            return text

        highlighted = text
        for term in query_terms:
            # Case-insensitive replacement with highlighting
            import re
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted = pattern.sub(f'<mark>{term}</mark>', highlighted)

        return highlighted


class SearchResult:
    """
    Container for a single search result with metadata.
    """

    def __init__(self, doc_id, document, score, matched_terms=None):
        """
        Initialize search result.

        Args:
            doc_id: Document identifier
            document: Full document data
            score: Relevance score
            matched_terms: List of matched query terms
        """
        self.doc_id = doc_id
        self.document = document
        self.score = score
        self.matched_terms = matched_terms or []

    @property
    def title(self):
        return self.document.get('title', 'Untitled')

    @property
    def authors(self):
        authors = self.document.get('authors', [])
        if isinstance(authors, list):
            return authors
        return [authors] if authors else []

    @property
    def year(self):
        return self.document.get('year', 'N/A')

    @property
    def abstract(self):
        return self.document.get('abstract', '')

    @property
    def publication_link(self):
        return self.document.get('publication_link', '')

    @property
    def author_profiles(self):
        return self.document.get('author_profiles', {})

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'doc_id': self.doc_id,
            'title': self.title,
            'authors': self.authors,
            'year': self.year,
            'abstract': self.abstract,
            'publication_link': self.publication_link,
            'author_profiles': self.author_profiles,
            'score': round(self.score, 4),
            'matched_terms': self.matched_terms
        }
