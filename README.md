# Fake News Classification Project

## Project Overview

This project focuses on building a text classification model to distinguish between real and fake news articles. Using natural language processing techniques and machine learning algorithms, we develop a classifier that can analyze news content and predict whether it's legitimate or fabricated.

## Dataset Structure

The dataset contains news articles with the following columns:

- **label**: 0 if the news is fake, 1 if the news is real
- **title**: The headline of the news article
- **text**: The full content of the article
- **subject**: The category or topic of the news
- **date**: The publication date of the article

## Project Structure

```
nlp_classification_fake_news/
├── README.md
├── requirements.txt
├── dataset/
│   ├── data.csv                    # Main training dataset
│   ├── validation_data.csv         # Validation dataset
│   └── processed_data.csv         # Preprocessed training data
├── notebooks/
│   ├── 01_data_exploration_preprocessing.ipynb
│   └── 02_model_training_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── text_preprocessing.py       # Text preprocessing utilities
│   └── model_utils.py             # Model training and evaluation utilities
├── models/
│   └── best_model_*.pkl           # Saved trained models
└── results/
    ├── validation_predictions.csv  # Final predictions
    ├── model_comparison.csv        # Model performance comparison
    └── model_summary_report.txt    # Detailed results report
```

## Installation and Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd nlp_classification_fake_news
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Exploration and Preprocessing

Open and run the first notebook:

```bash
jupyter notebook notebooks/01_data_exploration_preprocessing.ipynb
```

This notebook will:

- Load and explore the dataset
- Perform data cleaning and preprocessing
- Apply text processing techniques (tokenization, stemming, etc.)
- Split data into training and testing sets

### 2. Model Training and Evaluation

Open and run the second notebook:

```bash
jupyter notebook notebooks/02_model_training_evaluation.ipynb
```

This notebook will:

- Train multiple classification models
- Compare model performances
- Select the best performing model
- Generate predictions for validation data
- Save results and models

## Models Implemented

The project implements and compares several machine learning algorithms:

1. **Logistic Regression** - Linear classifier with TF-IDF features
2. **Random Forest** - Ensemble method with multiple decision trees
3. **Naive Bayes** - Probabilistic classifier ideal for text classification
4. **Support Vector Machine (SVM)** - Kernel-based classifier for complex patterns

## Text Processing Pipeline

1. **Text Cleaning**: Remove URLs, emails, special characters
2. **Normalization**: Convert to lowercase, handle whitespace
3. **Tokenization**: Split text into individual words
4. **Stopword Removal**: Remove common words without semantic meaning
5. **Stemming/Lemmatization**: Reduce words to their base forms
6. **Feature Extraction**: Convert text to numerical features using TF-IDF

## Evaluation Metrics

Models are evaluated using multiple metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Ability to avoid false positives
- **Recall**: Ability to find all positive instances
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the receiver operating characteristic curve

## Results

The trained models achieve the following performance (example):

- Best Model: Logistic Regression
- Accuracy: ~95%
- F1-Score: ~94%
- Precision: ~93%
- Recall: ~95%

_Note: Actual results will vary based on the dataset and training process._

## Deliverables

1. **Python Code**: Well-documented notebooks and utility modules
2. **Predictions**: CSV file with predicted labels for validation data
3. **Model Performance**: Comprehensive evaluation and comparison
4. **Accuracy Estimation**: Expected performance on new data

## Key Features

- **Modular Design**: Separate utilities for preprocessing and modeling
- **Multiple Models**: Comparison of different algorithms
- **Comprehensive Evaluation**: Detailed metrics and visualizations
- **Reproducible Results**: Fixed random seeds for consistency
- **Production Ready**: Saved models for future predictions

## Future Improvements

- Implement deep learning models (LSTM, BERT)
- Add feature engineering (n-grams, word embeddings)
- Perform hyperparameter tuning
- Add cross-validation for robust evaluation
- Implement ensemble methods

## License

This project is developed for educational purposes as part of the Ironhack Data Science bootcamp.

## Contributors

- [Your Name]
- Project developed as part of Ironhack NLP Classification assignment
