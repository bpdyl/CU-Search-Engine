"""
Inverted Index Implementation

Builds and maintains an inverted index for efficient document retrieval.
Supports field-based indexing with configurable weights.
"""
import os
import pickle
import math
from collections import defaultdict
from datetime import datetime

from .preprocessor import TextPreprocessor
import config


class InvertedIndex:
    """
    Inverted Index data structure for efficient text retrieval.

    Structure:
    - index: {term: [(doc_id, term_freq, field), ...]}
    - documents: {doc_id: document_data}
    - doc_lengths: {doc_id: document_length}
    """

    def __init__(self, field_weights=None):
        """
        Initialize the inverted index.

        Args:
            field_weights: Dictionary mapping field names to their weights
        """
        self.index = defaultdict(list)  # term -> [(doc_id, freq, field)]
        self.documents = {}  # doc_id -> full document data
        self.doc_lengths = {}  # doc_id -> total terms
        self.doc_field_lengths = {}  # doc_id -> {field: length}
        self.total_docs = 0
        self.avg_doc_length = 0

        self.field_weights = field_weights or config.FIELD_WEIGHTS
        self.preprocessor = TextPreprocessor(
            use_stemming=True,
            use_lemmatization=True,
            expand_synonyms=False
        )

        # Statistics
        self.term_doc_freq = defaultdict(int)  # term -> number of docs containing term
        self.created_at = None
        self.last_updated = None

    def add_document(self, doc_id, doc_data):
        """
        Add a document to the index.

        Args:
            doc_id: Unique document identifier
            doc_data: Dictionary containing document fields:
                - title: Publication title
                - authors: List of author names
                - year: Publication year
                - abstract: Abstract text
                - keywords: List of keywords
                - publication_link: URL to publication
                - author_profiles: Dict of author -> profile URL
        """
        self.documents[doc_id] = doc_data
        self.doc_field_lengths[doc_id] = {}

        # Define searchable fields
        searchable_fields = {
            'title': doc_data.get('title', ''),
            'authors': self._normalize_authors(doc_data.get('authors', [])),
            'year': str(doc_data.get('year', '')),
            'abstract': doc_data.get('abstract', ''),
            'keywords': self._normalize_keywords(doc_data.get('keywords', []))
        }

        total_terms = 0
        indexed_terms = set()  # Track terms for document frequency

        # Index each field
        for field, text in searchable_fields.items():
            if not text:
                continue

            # Preprocess text
            tokens = self.preprocessor.preprocess_for_indexing(text)
            self.doc_field_lengths[doc_id][field] = len(tokens)
            total_terms += len(tokens)

            # Get field weight
            weight = self.field_weights.get(field, 1.0)

            # Count term frequencies in this field
            term_freqs = defaultdict(int)
            for token in tokens:
                term_freqs[token] += 1
                indexed_terms.add(token)

            # Add to index
            for term, freq in term_freqs.items():
                # Check if document already has an entry for this term
                existing_idx = None
                for i, (did, _, _) in enumerate(self.index[term]):
                    if did == doc_id:
                        existing_idx = i
                        break

                if existing_idx is not None:
                    # Update existing entry
                    old_did, old_freq, old_field = self.index[term][existing_idx]
                    self.index[term][existing_idx] = (
                        doc_id,
                        old_freq + (freq * weight),
                        f"{old_field},{field}"
                    )
                else:
                    # Add new entry
                    self.index[term].append((doc_id, freq * weight, field))

        # Update term document frequencies
        for term in indexed_terms:
            self.term_doc_freq[term] += 1

        # Store document length
        self.doc_lengths[doc_id] = total_terms
        self.total_docs += 1

        # Update average document length
        self._update_avg_doc_length()

    def _normalize_authors(self, authors):
        """Convert authors list to searchable string."""
        if isinstance(authors, list):
            return ' '.join(authors)
        return str(authors)

    def _normalize_keywords(self, keywords):
        """Convert keywords list to searchable string."""
        if isinstance(keywords, list):
            return ' '.join(keywords)
        return str(keywords)

    def _update_avg_doc_length(self):
        """Update average document length."""
        if self.total_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs

    def get_document(self, doc_id):
        """
        Retrieve a document by ID.

        Args:
            doc_id: Document identifier

        Returns:
            Document data dictionary or None
        """
        return self.documents.get(doc_id)

    def get_postings(self, term):
        """
        Get postings list for a term.

        Args:
            term: Search term

        Returns:
            List of (doc_id, frequency, field) tuples
        """
        # Preprocess term
        processed = self.preprocessor.preprocess_for_indexing(term)
        if not processed:
            return []

        term = processed[0]
        return self.index.get(term, [])

    def get_document_frequency(self, term):
        """
        Get the number of documents containing a term.

        Args:
            term: Search term

        Returns:
            Document frequency count
        """
        processed = self.preprocessor.preprocess_for_indexing(term)
        if not processed:
            return 0

        return self.term_doc_freq.get(processed[0], 0)

    def get_idf(self, term):
        """
        Calculate Inverse Document Frequency for a term.

        Args:
            term: Search term

        Returns:
            IDF value
        """
        df = self.get_document_frequency(term)
        if df == 0:
            return 0

        return math.log((self.total_docs + 1) / (df + 1)) + 1

    def search_term(self, term):
        """
        Search for documents containing a term.

        Args:
            term: Search term

        Returns:
            List of (doc_id, document_data, score) tuples
        """
        postings = self.get_postings(term)
        idf = self.get_idf(term)

        results = []
        for doc_id, freq, field in postings:
            if doc_id in self.documents:
                score = freq * idf
                results.append((doc_id, self.documents[doc_id], score))

        return sorted(results, key=lambda x: x[2], reverse=True)

    def build_from_publications(self, publications):
        """
        Build index from a list of publications.

        Args:
            publications: List of publication dictionaries
        """
        self.clear()
        self.created_at = datetime.now().isoformat()

        for idx, pub in enumerate(publications):
            self.add_document(idx, pub)

        self.last_updated = datetime.now().isoformat()

    def clear(self):
        """Clear the entire index."""
        self.index.clear()
        self.documents.clear()
        self.doc_lengths.clear()
        self.doc_field_lengths.clear()
        self.term_doc_freq.clear()
        self.total_docs = 0
        self.avg_doc_length = 0

    def get_all_terms(self):
        """
        Get all indexed terms.

        Returns:
            Set of all terms in the index
        """
        return set(self.index.keys())

    def get_statistics(self):
        """
        Get index statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            'total_documents': self.total_docs,
            'total_terms': len(self.index),
            'average_doc_length': round(self.avg_doc_length, 2),
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'field_weights': self.field_weights
        }

    def save(self, filepath=None):
        """
        Save index to file.

        Args:
            filepath: Path to save file (default from config)
        """
        if filepath is None:
            filepath = config.INDEX_FILE

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = {
            'index': dict(self.index),
            'documents': self.documents,
            'doc_lengths': self.doc_lengths,
            'doc_field_lengths': self.doc_field_lengths,
            'term_doc_freq': dict(self.term_doc_freq),
            'total_docs': self.total_docs,
            'avg_doc_length': self.avg_doc_length,
            'field_weights': self.field_weights,
            'created_at': self.created_at,
            'last_updated': datetime.now().isoformat()
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath=None):
        """
        Load index from file.

        Args:
            filepath: Path to index file (default from config)

        Returns:
            True if loaded successfully, False otherwise
        """
        if filepath is None:
            filepath = config.INDEX_FILE

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            self.index = defaultdict(list, data.get('index', {}))
            self.documents = data.get('documents', {})
            self.doc_lengths = data.get('doc_lengths', {})
            self.doc_field_lengths = data.get('doc_field_lengths', {})
            self.term_doc_freq = defaultdict(int, data.get('term_doc_freq', {}))
            self.total_docs = data.get('total_docs', 0)
            self.avg_doc_length = data.get('avg_doc_length', 0)
            self.field_weights = data.get('field_weights', config.FIELD_WEIGHTS)
            self.created_at = data.get('created_at')
            self.last_updated = data.get('last_updated')

            return True

        except FileNotFoundError:
            return False
        except Exception as e:
            print(f"Error loading index: {e}")
            return False

    def __len__(self):
        """Return number of documents in index."""
        return self.total_docs

    def __contains__(self, term):
        """Check if term is in index."""
        processed = self.preprocessor.preprocess_for_indexing(term)
        if not processed:
            return False
        return processed[0] in self.index
