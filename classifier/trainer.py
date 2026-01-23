"""
Document Classification Trainer

Trains a Naive Bayes classifier for categorizing documents
into Business, Entertainment, and Health categories.
"""
import os
import json
import pickle
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import config
from indexer.preprocessor import TextPreprocessor


class ClassifierTrainer:
    """
    Trainer for the document classification model.
    Uses Naive Bayes with TF-IDF features.
    """

    def __init__(self):
        """Initialize the trainer."""
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.classifier = MultinomialNB(alpha=0.1)
        self.preprocessor = TextPreprocessor(
            use_stemming=False,
            use_lemmatization=True
        )
        self.categories = config.CLASSIFICATION_CATEGORIES
        self.is_trained = False
        self.training_stats = {}

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

    def load_training_data(self, filepath=None):
        """
        Load training data from JSON file.

        Expected format:
        [
            {"text": "...", "category": "Business"},
            {"text": "...", "category": "Entertainment"},
            ...
        ]

        Args:
            filepath: Path to training data file

        Returns:
            Tuple of (texts, labels)
        """
        if filepath is None:
            filepath = os.path.join(
                config.CLASSIFICATION_DATA_DIR,
                'labeled_articles.json'
            )

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            texts = []
            labels = []

            for item in data:
                text = item.get('text', '')
                category = item.get('category', '')

                if text and category in self.categories:
                    texts.append(text)
                    labels.append(category)

            return texts, labels

        except FileNotFoundError:
            print(f"Training data not found: {filepath}")
            return [], []
        except json.JSONDecodeError as e:
            print(f"Error parsing training data: {e}")
            return [], []

    def train(self, texts, labels, test_size=0.2):
        """
        Train the classifier.

        Args:
            texts: List of document texts
            labels: List of category labels
            test_size: Fraction of data for testing

        Returns:
            Dictionary of training statistics
        """
        if len(texts) < 10:
            raise ValueError("Insufficient training data (minimum 10 samples)")

        # Preprocess texts
        processed_texts = [self.preprocess_text(t) for t in texts]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels,
            test_size=test_size,
            random_state=42,
            stratify=labels
        )

        # Vectorize
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Train classifier
        self.classifier.fit(X_train_vec, y_train)

        # Evaluate
        y_pred = self.classifier.predict(X_test_vec)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred, labels=self.categories)
        class_report = classification_report(
            y_test, y_pred,
            target_names=self.categories,
            output_dict=True
        )

        # Cross-validation
        X_all_vec = self.vectorizer.transform(processed_texts)
        cv_scores = cross_val_score(self.classifier, X_all_vec, labels, cv=5)

        self.is_trained = True
        self.training_stats = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'total_samples': len(texts),
            'trained_at': datetime.now().isoformat()
        }

        return self.training_stats

    def save_model(self, model_path=None, vectorizer_path=None):
        """
        Save trained model and vectorizer.

        Args:
            model_path: Path to save classifier model
            vectorizer_path: Path to save vectorizer
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        if model_path is None:
            model_path = config.CLASSIFIER_MODEL_FILE
        if vectorizer_path is None:
            vectorizer_path = config.VECTORIZER_FILE

        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save classifier
        with open(model_path, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'categories': self.categories,
                'stats': self.training_stats
            }, f)

        # Save vectorizer
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")

    def get_training_report(self):
        """
        Get formatted training report.

        Returns:
            String report of training results
        """
        if not self.training_stats:
            return "No training statistics available"

        stats = self.training_stats
        report = []

        report.append("=" * 50)
        report.append("DOCUMENT CLASSIFIER TRAINING REPORT")
        report.append("=" * 50)
        report.append("")
        report.append(f"Training Date: {stats.get('trained_at', 'N/A')}")
        report.append(f"Total Samples: {stats.get('total_samples', 0)}")
        report.append(f"Training Set Size: {stats.get('train_size', 0)}")
        report.append(f"Test Set Size: {stats.get('test_size', 0)}")
        report.append("")
        report.append(f"Test Accuracy: {stats.get('accuracy', 0):.4f}")
        report.append(f"Cross-Validation Mean: {stats.get('cv_mean', 0):.4f}")
        report.append(f"Cross-Validation Std: {stats.get('cv_std', 0):.4f}")
        report.append("")
        report.append("Confusion Matrix:")

        # Format confusion matrix
        conf_matrix = stats.get('confusion_matrix', [])
        if conf_matrix:
            header = "           " + "  ".join(f"{c[:8]:>8}" for c in self.categories)
            report.append(header)
            for i, row in enumerate(conf_matrix):
                row_str = f"{self.categories[i][:10]:<10} " + "  ".join(f"{v:>8}" for v in row)
                report.append(row_str)

        report.append("")
        report.append("Per-Class Metrics:")

        class_report = stats.get('classification_report', {})
        for category in self.categories:
            if category in class_report:
                metrics = class_report[category]
                report.append(f"  {category}:")
                report.append(f"    Precision: {metrics.get('precision', 0):.4f}")
                report.append(f"    Recall: {metrics.get('recall', 0):.4f}")
                report.append(f"    F1-Score: {metrics.get('f1-score', 0):.4f}")

        report.append("=" * 50)

        return "\n".join(report)


def create_sample_training_data():
    """
    Create comprehensive training data for the classifier.
    Uses the expanded dataset from training_data module (400+ documents).

    Includes:
    - Long-form articles (news article style)
    - Medium-length summaries
    - Short-form queries and phrases (TV shows, movies, business terms, health topics)

    Data Sources Attribution:
    - Content style based on BBC News (https://www.bbc.com/news)
    - Content style based on Reuters (https://www.reuters.com)
    - Content style based on The Guardian (https://www.theguardian.com)
    - Content style based on CNN (https://www.cnn.com)

    Returns:
        List of training samples (400+ documents)
    """
    from .training_data import get_training_data
    return get_training_data()


def save_sample_training_data():
    """Save comprehensive training data to file."""
    samples = create_sample_training_data()

    filepath = os.path.join(
        config.CLASSIFICATION_DATA_DIR,
        'labeled_articles.json'
    )

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2)

    print(f"\n{'='*50}")
    print("TRAINING DATA SAVED")
    print(f"{'='*50}")
    print(f"File: {filepath}")
    print(f"Total samples: {len(samples)}")
    print()

    # Count by category
    print("Samples per category:")
    category_counts = {}
    for sample in samples:
        cat = sample['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1

    for cat, count in sorted(category_counts.items()):
        print(f"  - {cat}: {count}")

    print(f"{'='*50}\n")

    return filepath
