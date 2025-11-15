# Simple Fake News Classification Results

Generated: 2025-11-15 12:32:14

## Dataset Information
- Total articles: 39,935
- Training samples: 31,948
- Test samples: 7,987

## Preprocessing
- Removed news sources: (reuters), (reuter), (ap), (afp) - case insensitive
- Removed standalone "reuters" and "reuter" as whole words - case insensitive
- Removed caps+colon patterns at start of text (e.g., "UNBELIEVABLE:")

## Model Configuration
- Algorithm: Logistic Regression
- TF-IDF Features: 1,000
- N-grams: (1, 1) - unigrams only
- Max features: 1000
- Regularization (C): 0.01 (strong regularization to reduce overfitting)

## Model Performance (Test Set)
- **Accuracy**: 0.9356 (93.56%)
- **Precision**: 0.9251
- **Recall**: 0.9483
- **F1-Score**: 0.9365

## Cross-Validation Performance (5-fold)
- **CV Accuracy**: 0.9343 ± 0.0022
- **CV Precision**: 0.9232 ± 0.0045
- **CV Recall**: 0.9476 ± 0.0012
- **CV F1-Score**: 0.9352 ± 0.0020

## Overfitting Analysis
- Test vs CV Accuracy Difference: 0.0014
- ✓ Model generalizes well

## Summary
The Logistic Regression classifier achieved 93.56% accuracy on the test set, with a balanced F1-score of 0.9365. Cross-validation shows 93.43% accuracy, indicating good generalization. The model uses enhanced preprocessing to remove news sources and caps+colon patterns (common in fake news), followed by standard NLP preprocessing (stemming, stopword removal).
