import os
import re
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
import traceback
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd

# Download NLTK data if needed
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# --- Configuration ---
CONFIG = {
    "CLASS_NAMES": ["Fake", "Real"],
    "LABEL_MAPPING": {0: "Fake", 1: "Real"}
}

# --- Text Preprocessing Function (matches notebook) ---
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def standard_preprocess(text):
    """Preprocess text to match training preprocessing."""
    if pd.isna(text) or text == "":
        return ""
    
    text_str = str(text)
    
    # Remove news sources (case-insensitive)
    text_str = re.sub(r'\(reuters\)|\(reuter\)|\(ap\)|\(afp\)', '', text_str, flags=re.IGNORECASE)
    text_str = re.sub(r'\breuters\b', '', text_str, flags=re.IGNORECASE)
    text_str = re.sub(r'\breuter\b', '', text_str, flags=re.IGNORECASE)
    
    # Remove caps+colon patterns at start
    text_str = re.sub(r'^[A-Z]{5,}:\s*', '', text_str)
    
    text_str = text_str.lower()
    text_str = re.sub(r'http\S+|www\S+|https\S+', '', text_str)
    text_str = re.sub(r'\S+@\S+', '', text_str)
    text_str = re.sub(r'\s+', ' ', text_str)
    
    try:
        tokens = word_tokenize(text_str)
    except:
        tokens = text_str.split()
    
    tokens = [token for token in tokens if token.lower() not in stop_words]
    tokens = [stemmer.stem(token) for token in tokens]
    tokens = [token for token in tokens if len(token) > 2]
    
    return ' '.join(tokens)

# --- 1. LOAD THE MODEL (Global Scope) ---
model = None
vectorizer = None
pipeline = None

try:
    # Try loading the pipeline first (easiest option)
    pipeline_path = 'model_pipeline.pkl'
    if os.path.exists(pipeline_path):
        pipeline = joblib.load(pipeline_path)
        print("--- Pipeline loaded successfully ---")
    else:
        # Fall back to loading model and vectorizer separately
        model_path = 'exported_model.pkl'
        vectorizer_path = 'tfidf_vectorizer.pkl'
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            print("--- Model and vectorizer loaded successfully ---")
        else:
            print(f"Error: Model files not found. Looking for {pipeline_path} or {model_path} and {vectorizer_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()
    model = None
    vectorizer = None
    pipeline = None

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Main Page Route ---
@app.route("/")
def home():
    """Serves the main index.html page."""
    return render_template("index.html")

# --- Text Preprocessing for Prediction ---
def preprocess_text_for_prediction(title, text):
    """
    Preprocess title and text, then combine them for prediction.
    Matches the training data format.
    """
    title_processed = standard_preprocess(title)
    text_processed = standard_preprocess(text)
    combined_text = title_processed + ' ' + text_processed
    return combined_text

# --- API Endpoint (Text Input via POST) ---
@app.route("/predict", methods=["POST"])
def predict():
    if pipeline is None and (model is None or vectorizer is None):
        return jsonify({"error": "Model is not loaded or failed to load."}), 500

    try:
        # Get text data from request
        if request.is_json:
            data = request.get_json()
            title = data.get('title', '')
            text = data.get('text', '')
        else:
            title = request.form.get('title', '')
            text = request.form.get('text', '')
        
        if not title and not text:
            return jsonify({"error": "No 'title' or 'text' provided in request"}), 400
        
        # Preprocess the text
        combined_text = preprocess_text_for_prediction(title, text)
        
        if not combined_text.strip():
            return jsonify({"error": "Text is empty after preprocessing"}), 400
        
        # Make prediction
        if pipeline:
            # Use pipeline (includes vectorization + prediction)
            prediction = pipeline.predict([combined_text])[0]
            probabilities = pipeline.predict_proba([combined_text])[0]
        else:
            # Use separate model and vectorizer
            text_vectorized = vectorizer.transform([combined_text])
            prediction = model.predict(text_vectorized)[0]
            probabilities = model.predict_proba(text_vectorized)[0]
        
        # Format response
        predicted_class = CONFIG["LABEL_MAPPING"][int(prediction)]
        confidence = float(probabilities[int(prediction)])
        
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": {
                "Fake": float(probabilities[0]),
                "Real": float(probabilities[1])
            },
            "predicted_label": int(prediction)
        })

    except Exception as e:
        print("="*50)
        print(f"ERROR: An internal Flask error occurred:")
        traceback.print_exc()
        print("="*50)
        return jsonify({
            "error": "An internal server error occurred.",
            "detail": str(e)
        }), 500

# --- Health Check Endpoint ---
@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    if pipeline is None and (model is None or vectorizer is None):
        return jsonify({"status": "unhealthy", "message": "Model not loaded"}), 503
    return jsonify({"status": "healthy", "message": "Model loaded and ready"}), 200

# Gunicorn/Cloud Run AND for local testing.
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
