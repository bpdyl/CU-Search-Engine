"""
Query Processor for Search Engine

Handles query parsing, preprocessing, and retrieval with ranking.
Supports partial matching, synonym expansion, and pagination.
"""
from collections import defaultdict
import heapq

from indexer.preprocessor import TextPreprocessor
from indexer.ranking import get_ranker, BM25Ranker
import config


class QueryProcessor:
    """
    Processes search queries and retrieves ranked results.
    Optimized for fast response times with pagination support.
    """

    def __init__(self, inverted_index, ranking_algorithm='tfidf'):
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
            expand_synonyms=False  # Disabled for performance - synonyms cause O(n) WordNet lookups
        )
        
        # Cache for partial matches (built lazily)
        self._partial_match_cache = {}
        self._cache_version = None

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

    def _get_partial_matches_optimized(self, term, all_terms):
        """
        Get partial matches with caching for better performance.
        
        Args:
            term: Query term to match
            all_terms: Set of all indexed terms
            
        Returns:
            List of matching terms
        """
        # Check if cache needs rebuilding (index changed)
        current_version = len(all_terms)
        if self._cache_version != current_version:
            self._partial_match_cache.clear()
            self._cache_version = current_version
        
        # Check cache
        if term in self._partial_match_cache:
            return self._partial_match_cache[term]
        
        # Find matches (limited to avoid excessive processing)
        matches = []
        match_limit = 50  # Limit partial matches per term
        
        for indexed_term in all_terms:
            if len(matches) >= match_limit:
                break
            if term == indexed_term:
                matches.append(indexed_term)
            elif indexed_term.startswith(term) or term.startswith(indexed_term):
                matches.append(indexed_term)
        
        # Cache result
        self._partial_match_cache[term] = matches
        return matches

    def search(self, query, limit=None, page=1, per_page=20, sort_by='relevance'):
        """
        Search for documents matching the query with pagination and sorting.

        Args:
            query: Search query string
            limit: Maximum number of results (default from config) - for backward compatibility
            page: Page number (1-indexed)
            per_page: Results per page
            sort_by: Sort order - 'relevance', 'year_desc', 'year_asc'

        Returns:
            Dictionary with results and pagination info:
            {
                'results': List of (doc_id, document_data, score) tuples,
                'total': Total number of matching documents,
                'page': Current page,
                'per_page': Results per page,
                'total_pages': Total number of pages
            }
        """
        # Handle backward compatibility - if limit is set, use old behavior
        if limit is not None:
            return self._search_simple(query, limit)
        
        if not query or not query.strip():
            return {'results': [], 'total': 0, 'page': 1, 'per_page': per_page, 'total_pages': 0}

        # Parse query
        query_terms, original_terms = self.parse_query(query)

        # Limit query terms to prevent performance degradation with very long queries
        MAX_QUERY_TERMS = 15
        if len(query_terms) > MAX_QUERY_TERMS:
            query_terms = query_terms[:MAX_QUERY_TERMS]

        if not query_terms:
            return {'results': [], 'total': 0, 'page': 1, 'per_page': per_page, 'total_pages': 0}

        # Calculate term frequencies in query
        query_term_freqs = defaultdict(int)
        for term in query_terms:
            query_term_freqs[term] += 1

        # Build posting lists cache and find candidate documents
        postings_cache = {}
        doc_term_matches = defaultdict(set)  # doc_id -> set of matched terms
        
        for term in query_terms:
            postings = self.index.get_postings(term)
            postings_cache[term] = {doc_id: (freq, field) for doc_id, freq, field in postings}
            for doc_id, _, _ in postings:
                doc_term_matches[doc_id].add(term)

        # Limited partial matching (only if few exact matches AND short queries)
        # Skip for long queries as they are already specific enough
        if len(doc_term_matches) < 100 and len(query_terms) <= 5:
            all_terms = self.index.get_all_terms()
            for term in query_terms:
                partial_matches = self._get_partial_matches_optimized(term, all_terms)
                for match in partial_matches:
                    if match not in postings_cache:
                        postings = self.index.index.get(match, [])
                        for doc_id, _, _ in postings:
                            doc_term_matches[doc_id].add(term)

        candidate_docs = set(doc_term_matches.keys())

        if not candidate_docs:
            return {'results': [], 'total': 0, 'page': 1, 'per_page': per_page, 'total_pages': 0}

        # Score documents efficiently
        scored_docs = []
        num_query_terms = len(query_terms)
        
        for doc_id in candidate_docs:
            doc = self.index.get_document(doc_id)
            if doc:
                score = self.ranker.score_document(doc_id, query_terms, query_term_freqs, postings_cache)

                # Boost score based on term coverage (use cached matches)
                if num_query_terms > 1:
                    matched_terms = len(doc_term_matches[doc_id])
                    coverage = matched_terms / num_query_terms
                    score *= (1 + coverage)

                scored_docs.append((score, doc_id, doc))

        # Sort based on sort_by parameter
        if sort_by == 'year_desc':
            # Sort by year descending, then by score
            scored_docs.sort(key=lambda x: (self._get_year_for_sort(x[2]), x[0]), reverse=True)
        elif sort_by == 'year_asc':
            # Sort by year ascending (oldest first), then by score descending within same year
            scored_docs.sort(key=lambda x: (self._get_year_for_sort(x[2]), -x[0]))
        else:
            # Default: sort by relevance score descending
            scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Pagination
        total = len(scored_docs)
        total_pages = (total + per_page - 1) // per_page if total > 0 else 0
        page = max(1, min(page, total_pages)) if total_pages > 0 else 1
        
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        # Extract results for current page
        results = [(doc_id, doc, score) for score, doc_id, doc in scored_docs[start_idx:end_idx]]

        return {
            'results': results,
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': total_pages
        }
    
    def _get_year_for_sort(self, doc):
        """
        Extract year from document for sorting.
        
        Args:
            doc: Document dictionary
            
        Returns:
            Year as integer, or 0 if not available
        """
        year = doc.get('year', 0)
        try:
            return int(year)
        except (ValueError, TypeError):
            return 0
    
    def _search_simple(self, query, limit):
        """
        Simple search without pagination (backward compatibility).
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of (doc_id, document_data, score) tuples
        """
        if not query or not query.strip():
            return []

        # Parse query
        query_terms, original_terms = self.parse_query(query)

        # Limit query terms to prevent performance degradation with very long queries
        MAX_QUERY_TERMS = 15
        if len(query_terms) > MAX_QUERY_TERMS:
            query_terms = query_terms[:MAX_QUERY_TERMS]

        if not query_terms:
            return []

        # Calculate term frequencies in query
        query_term_freqs = defaultdict(int)
        for term in query_terms:
            query_term_freqs[term] += 1

        # Build posting lists cache and find candidate documents
        postings_cache = {}
        doc_term_matches = defaultdict(set)
        
        for term in query_terms:
            postings = self.index.get_postings(term)
            postings_cache[term] = {doc_id: (freq, field) for doc_id, freq, field in postings}
            for doc_id, _, _ in postings:
                doc_term_matches[doc_id].add(term)

        # Limited partial matching (only for short queries)
        if len(doc_term_matches) < 100 and len(query_terms) <= 5:
            all_terms = self.index.get_all_terms()
            for term in query_terms:
                partial_matches = self._get_partial_matches_optimized(term, all_terms)
                for match in partial_matches:
                    if match not in postings_cache:
                        postings = self.index.index.get(match, [])
                        for doc_id, _, _ in postings:
                            doc_term_matches[doc_id].add(term)

        candidate_docs = set(doc_term_matches.keys())

        if not candidate_docs:
            return []

        # Use heap for top-k selection (more efficient than sorting all)
        num_query_terms = len(query_terms)
        top_k = []
        
        for doc_id in candidate_docs:
            doc = self.index.get_document(doc_id)
            if doc:
                score = self.ranker.score_document(doc_id, query_terms, query_term_freqs, postings_cache)

                if num_query_terms > 1:
                    matched_terms = len(doc_term_matches[doc_id])
                    coverage = matched_terms / num_query_terms
                    score *= (1 + coverage)

                if len(top_k) < limit:
                    heapq.heappush(top_k, (score, doc_id, doc))
                elif score > top_k[0][0]:
                    heapq.heapreplace(top_k, (score, doc_id, doc))

        # Extract and sort results
        results = [(doc_id, doc, score) for score, doc_id, doc in top_k]
        results.sort(key=lambda x: x[2], reverse=True)

        return results

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
