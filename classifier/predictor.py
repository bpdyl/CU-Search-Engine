"""
Document Classification Predictor

Loads trained model and classifies new documents.
"""
import os
import pickle

import config
from indexer.preprocessor import TextPreprocessor


class DocumentClassifier:
    """
    Classifier for predicting document categories.
    """

    def __init__(self):
        """Initialize the classifier."""
        self.classifier = None
        self.vectorizer = None
        self.categories = config.CLASSIFICATION_CATEGORIES
        self.is_loaded = False
        self.model_stats = {}
        self.preprocessor = TextPreprocessor(
            use_stemming=False,
            use_lemmatization=True
        )

    def load_model(self, model_path=None, vectorizer_path=None):
        """
        Load trained model and vectorizer.

        Args:
            model_path: Path to classifier model
            vectorizer_path: Path to vectorizer

        Returns:
            True if loaded successfully, False otherwise
        """
        if model_path is None:
            model_path = config.CLASSIFIER_MODEL_FILE
        if vectorizer_path is None:
            vectorizer_path = config.VECTORIZER_FILE

        try:
            # Load classifier
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.classifier = model_data['classifier']
                self.categories = model_data.get('categories', self.categories)
                self.model_stats = model_data.get('stats', {})

            # Load vectorizer
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)

            self.is_loaded = True
            return True

        except FileNotFoundError as e:
            print(f"Model file not found: {e}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def preprocess_text(self, text):
        """
        Preprocess text for classification.

        Args:
            text: Raw text

        Returns:
            Preprocessed text string
        """
        tokens = self.preprocessor.preprocess(text)
        return ' '.join(tokens)

    def classify(self, text):
        """
        Classify a document.

        Args:
            text: Document text to classify

        Returns:
            Dictionary with predicted category and probabilities
        """
        if not self.is_loaded:
            if not self.load_model():
                return {
                    'error': 'Model not loaded',
                    'category': None,
                    'confidence': 0,
                    'probabilities': {}
                }

        if not text or not text.strip():
            return {
                'error': 'Empty text',
                'category': None,
                'confidence': 0,
                'probabilities': {}
            }

        # Preprocess
        processed = self.preprocess_text(text)

        # Vectorize
        X = self.vectorizer.transform([processed])

        # Predict
        prediction = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]

        # Create probability dictionary
        prob_dict = {}
        for i, category in enumerate(self.classifier.classes_):
            prob_dict[category] = round(float(probabilities[i]), 4)

        # Get confidence (probability of predicted class)
        predicted_idx = list(self.classifier.classes_).index(prediction)
        confidence = probabilities[predicted_idx]

        return {
            'category': prediction,
            'confidence': round(float(confidence), 4),
            'probabilities': prob_dict,
            'text_preview': text[:200] + '...' if len(text) > 200 else text
        }

    def classify_batch(self, texts):
        """
        Classify multiple documents.

        Args:
            texts: List of document texts

        Returns:
            List of classification results
        """
        return [self.classify(text) for text in texts]

    def get_model_info(self):
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if not self.is_loaded:
            return {'status': 'Model not loaded'}

        return {
            'status': 'Model loaded',
            'categories': self.categories,
            'accuracy': self.model_stats.get('accuracy', 'N/A'),
            'cv_mean': self.model_stats.get('cv_mean', 'N/A'),
            'trained_at': self.model_stats.get('trained_at', 'N/A'),
            'total_samples': self.model_stats.get('total_samples', 'N/A')
        }

    def is_ready(self):
        """Check if classifier is ready to use."""
        return self.is_loaded


def get_classifier():
    """
    Get a ready-to-use classifier instance.

    Returns:
        DocumentClassifier instance (may not be loaded)
    """
    classifier = DocumentClassifier()
    classifier.load_model()
    return classifier
