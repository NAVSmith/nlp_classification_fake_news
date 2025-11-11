# Fake News Classification - Quick Start Guide

## Getting Started

1. **Setup Environment**

   ```bash
   python setup.py
   ```

   Or manually:

   ```bash
   pip install -r requirements.txt
   ```

2. **Add Your Data**

   - Place `data.csv` in the `dataset/` folder
   - Place `validation_data.csv` in the `dataset/` folder
   - Format should match the sample files provided

3. **Run Notebooks**
   ```bash
   jupyter notebook
   ```
   Then open and run:
   - `notebooks/01_data_exploration_preprocessing.ipynb`
   - `notebooks/02_model_training_evaluation.ipynb`

## Expected Data Format

### Training Data (data.csv)

```csv
label,title,text,subject,date
0,"Fake headline","Fake article content","Politics","2023-01-01"
1,"Real headline","Real article content","Science","2023-01-01"
```

### Validation Data (validation_data.csv)

```csv
label,title,text,subject,date
2,"Unknown headline","Article to classify","Technology","2023-01-01"
```

_Note: Label 2 will be replaced with 0 (fake) or 1 (real)_

## Key Components

### Notebooks

- **01_data_exploration_preprocessing.ipynb**: Data loading, cleaning, and preprocessing
- **02_model_training_evaluation.ipynb**: Model training, evaluation, and predictions

### Source Code

- **src/text_preprocessing.py**: Text cleaning and processing utilities
- **src/model_utils.py**: Machine learning model utilities

### Output Files

- **results/validation_predictions.csv**: Final predictions
- **results/model_comparison.csv**: Model performance comparison
- **results/model_summary_report.txt**: Detailed results
- **models/best*model*\*.pkl**: Saved trained model

## Workflow

1. **Data Exploration** → Load and understand the dataset
2. **Preprocessing** → Clean and prepare text data
3. **Feature Engineering** → Convert text to numerical features
4. **Model Training** → Train multiple classifiers
5. **Evaluation** → Compare model performances
6. **Prediction** → Generate predictions for validation data
7. **Export** → Save results and models

## Tips for Success

- Ensure your data follows the expected format
- Run notebooks in order (01 before 02)
- Check data quality before training
- Monitor model performance metrics
- Save your work frequently

## Troubleshooting

- **Import errors**: Run `pip install -r requirements.txt`
- **NLTK errors**: Run `python -c "import nltk; nltk.download('all')"`
- **Data errors**: Check CSV format and column names
- **Memory errors**: Reduce max_features in TfidfVectorizer

## Project Structure Reference

```
📁 nlp_classification_fake_news/
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 setup.py
├── 📄 QUICK_START.md
├── 📁 dataset/
│   ├── 📄 data.csv (your training data)
│   ├── 📄 validation_data.csv (your validation data)
│   ├── 📄 sample_data.csv (example format)
│   └── 📄 processed_data.csv (generated)
├── 📁 notebooks/
│   ├── 📄 01_data_exploration_preprocessing.ipynb
│   └── 📄 02_model_training_evaluation.ipynb
├── 📁 src/
│   ├── 📄 __init__.py
│   ├── 📄 text_preprocessing.py
│   └── 📄 model_utils.py
├── 📁 models/
│   └── 📄 best_model_*.pkl (generated)
└── 📁 results/
    ├── 📄 validation_predictions.csv (generated)
    ├── 📄 model_comparison.csv (generated)
    └── 📄 model_summary_report.txt (generated)
```
