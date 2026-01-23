"""
Document Classification module using Naive Bayes
"""
from .trainer import ClassifierTrainer
from .predictor import DocumentClassifier

__all__ = ['ClassifierTrainer', 'DocumentClassifier']
