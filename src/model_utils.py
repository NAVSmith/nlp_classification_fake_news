"""
Model utilities for Fake News Classification

This module contains functions for training, evaluating, and comparing
different machine learning models for fake news classification.
"""

import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.pipeline import Pipeline

class FakeNewsClassifier:
    """A comprehensive class for fake news classification."""
    
    def __init__(self):
        """Initialize the classifier with default models."""
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def create_models(self):
        """Create different machine learning models with preprocessing pipelines."""
        self.models = {
            'Logistic Regression': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ]),
            'Random Forest': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ]),
            'Naive Bayes': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
                ('classifier', MultinomialNB())
            ]),
            'SVM': Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
                ('classifier', SVC(random_state=42, probability=True))
            ])
        }
        return self.models
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate all models.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training labels
            y_test: Testing labels
            
        Returns:
            dict: Results for all models
        """
        if not self.models:
            self.create_models()
            
        self.results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Fit the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Store results
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print("-" * 30)
        
        return self.results
    
    def get_best_model(self, metric='f1'):
        """
        Get the best performing model based on specified metric.
        
        Args:
            metric (str): Metric to use for selection ('accuracy', 'f1', 'precision', 'recall', 'roc_auc')
            
        Returns:
            tuple: (best_model_name, best_model)
        """
        if not self.results:
            raise ValueError("No results available. Please train models first.")
        
        best_score = 0
        best_name = None
        
        for name, result in self.results.items():
            if result[metric] > best_score:
                best_score = result[metric]
                best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.results[best_name]['model']
        
        return best_name, self.best_model
    
    def get_comparison_dataframe(self):
        """
        Create a comparison dataframe of all model results.
        
        Returns:
            pd.DataFrame: Comparison of all models
        """
        if not self.results:
            raise ValueError("No results available. Please train models first.")
        
        comparison_data = {
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[model]['accuracy'] for model in self.results.keys()],
            'Precision': [self.results[model]['precision'] for model in self.results.keys()],
            'Recall': [self.results[model]['recall'] for model in self.results.keys()],
            'F1-Score': [self.results[model]['f1'] for model in self.results.keys()],
            'ROC AUC': [self.results[model]['roc_auc'] for model in self.results.keys()]
        }
        
        return pd.DataFrame(comparison_data)
    
    def predict_validation_data(self, validation_text):
        """
        Predict labels for validation data using the best model.
        
        Args:
            validation_text: Text data to predict
            
        Returns:
            tuple: (predictions, probabilities)
        """
        if self.best_model is None:
            raise ValueError("No best model available. Please train and select best model first.")
        
        predictions = self.best_model.predict(validation_text)
        probabilities = self.best_model.predict_proba(validation_text)
        
        return predictions, probabilities
    
    def save_model(self, filename):
        """Save the best model to a file."""
        if self.best_model is None:
            raise ValueError("No best model available to save.")
        
        joblib.dump(self.best_model, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load a model from a file."""
        self.best_model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return self.best_model

def evaluate_model_detailed(y_true, y_pred, model_name="Model"):
    """
    Provide detailed evaluation of a model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        
    Returns:
        dict: Detailed evaluation metrics
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'confusion_matrix': cm
    }
    
    return results