"""
Text Preprocessing Pipeline using NLTK

Implements:
- Tokenization
- Lowercasing
- Stopword removal
- Stemming (Porter Stemmer)
- Lemmatization (WordNet)
- Synonym expansion (WordNet)
"""
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK resources"""
    resources = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng',
        'omw-1.4'
    ]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {resource}: {e}")


class TextPreprocessor:
    """
    Text preprocessing class that provides various NLP operations
    for preparing text for indexing and searching.
    """

    def __init__(self, use_stemming=True, use_lemmatization=True,
                 expand_synonyms=False):
        """
        Initialize the preprocessor.

        Args:
            use_stemming: Apply Porter Stemmer
            use_lemmatization: Apply WordNet Lemmatizer
            expand_synonyms: Expand terms with synonyms from WordNet
        """
        # Download NLTK data on initialization
        download_nltk_data()

        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.expand_synonyms = expand_synonyms

        # Initialize tools
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        # Load stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback stopwords if NLTK data not available
            self.stop_words = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
                'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
                'or', 'that', 'the', 'to', 'was', 'will', 'with', 'this',
                'but', 'they', 'have', 'had', 'what', 'when', 'where',
                'who', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
                'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                'only', 'same', 'so', 'than', 'too', 'very', 'can', 'just',
                'should', 'now', 'i', 'we', 'you', 'he', 'she', 'them',
                'their', 'there', 'here', 'about', 'after', 'before',
                'above', 'below', 'between', 'during', 'through', 'into'
            }

    def clean_text(self, text):
        """
        Clean text by removing special characters and extra whitespace.

        Args:
            text: Input text string

        Returns:
            Cleaned text string
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove numbers (optional - keep for year matching)
        # text = re.sub(r'\d+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize(self, text):
        """
        Tokenize text into words.

        Args:
            text: Input text string

        Returns:
            List of tokens
        """
        if not text:
            return []

        try:
            tokens = word_tokenize(text)
        except:
            # Fallback to simple split if NLTK tokenizer fails
            tokens = text.split()

        return tokens

    def remove_stopwords(self, tokens):
        """
        Remove stopwords from token list.

        Args:
            tokens: List of tokens

        Returns:
            Filtered list of tokens
        """
        return [t for t in tokens if t.lower() not in self.stop_words and len(t) > 1]

    def stem(self, tokens):
        """
        Apply Porter Stemmer to tokens.

        Args:
            tokens: List of tokens

        Returns:
            List of stemmed tokens
        """
        return [self.stemmer.stem(token) for token in tokens]

    def get_wordnet_pos(self, tag):
        """
        Map POS tag to WordNet POS tag.

        Args:
            tag: NLTK POS tag

        Returns:
            WordNet POS tag
        """
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def lemmatize(self, tokens):
        """
        Apply WordNet Lemmatizer to tokens with POS tagging.

        Args:
            tokens: List of tokens

        Returns:
            List of lemmatized tokens
        """
        try:
            # Get POS tags
            pos_tags = pos_tag(tokens)

            # Lemmatize with POS
            lemmatized = []
            for word, tag in pos_tags:
                pos = self.get_wordnet_pos(tag)
                lemmatized.append(self.lemmatizer.lemmatize(word, pos))

            return lemmatized
        except:
            # Fallback without POS tagging
            return [self.lemmatizer.lemmatize(token) for token in tokens]

    def get_synonyms(self, word, max_synonyms=3):
        """
        Get synonyms for a word from WordNet.

        Args:
            word: Input word
            max_synonyms: Maximum number of synonyms to return

        Returns:
            List of synonyms
        """
        synonyms = set()

        try:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().lower().replace('_', ' ')
                    if synonym != word.lower():
                        synonyms.add(synonym)
                        if len(synonyms) >= max_synonyms:
                            return list(synonyms)
        except:
            pass

        return list(synonyms)

    def expand_with_synonyms(self, tokens, max_synonyms=2):
        """
        Expand token list with synonyms.

        Args:
            tokens: List of tokens
            max_synonyms: Maximum synonyms per token

        Returns:
            Expanded list of tokens
        """
        expanded = list(tokens)

        for token in tokens:
            synonyms = self.get_synonyms(token, max_synonyms)
            expanded.extend(synonyms)

        return expanded

    def preprocess(self, text, for_query=False):
        """
        Full preprocessing pipeline.

        Args:
            text: Input text string
            for_query: If True, expand with synonyms for better recall

        Returns:
            List of processed tokens
        """
        # Clean text
        cleaned = self.clean_text(text)

        # Tokenize
        tokens = self.tokenize(cleaned)

        # Remove stopwords
        tokens = self.remove_stopwords(tokens)

        # Apply lemmatization if enabled
        if self.use_lemmatization:
            tokens = self.lemmatize(tokens)

        # Apply stemming if enabled
        if self.use_stemming:
            tokens = self.stem(tokens)

        # Expand with synonyms for queries
        if for_query and self.expand_synonyms:
            tokens = self.expand_with_synonyms(tokens)

        return tokens

    def preprocess_for_indexing(self, text):
        """
        Preprocess text for indexing (no synonym expansion).

        Args:
            text: Input text string

        Returns:
            List of processed tokens
        """
        return self.preprocess(text, for_query=False)

    def preprocess_for_query(self, text):
        """
        Preprocess text for querying (with optional synonym expansion).

        Args:
            text: Input text string

        Returns:
            List of processed tokens
        """
        return self.preprocess(text, for_query=True)

    def get_partial_matches(self, query_token, indexed_tokens):
        """
        Find partial matches for a query token in indexed tokens.
        Supports partial phrase matching (e.g., "machine" matches "machine learning")

        Args:
            query_token: Query token to match
            indexed_tokens: Set of indexed tokens

        Returns:
            List of matching tokens
        """
        matches = []

        for token in indexed_tokens:
            # Exact match
            if query_token == token:
                matches.append(token)
            # Partial match (query token is prefix)
            elif token.startswith(query_token):
                matches.append(token)
            # Partial match (query token contains indexed token)
            elif query_token.startswith(token):
                matches.append(token)

        return matches


# Convenience function for quick preprocessing
def preprocess_text(text, use_stemming=True, use_lemmatization=True):
    """
    Quick preprocessing function.

    Args:
        text: Input text
        use_stemming: Apply stemming
        use_lemmatization: Apply lemmatization

    Returns:
        List of processed tokens
    """
    preprocessor = TextPreprocessor(
        use_stemming=use_stemming,
        use_lemmatization=use_lemmatization
    )
    return preprocessor.preprocess(text)
