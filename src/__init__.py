"""
Fake News Classification Package

This package contains utilities for text preprocessing and machine learning
model training for fake news classification.
"""

__version__ = "1.0.0"
__author__ = "Ironhack Student"

from .text_preprocessing import TextPreprocessor, get_text_statistics
from .model_utils import FakeNewsClassifier, evaluate_model_detailed

__all__ = [
    'TextPreprocessor',
    'get_text_statistics', 
    'FakeNewsClassifier',
    'evaluate_model_detailed'
]