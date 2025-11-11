"""
Text Preprocessing Utilities for Fake News Classification

This module contains functions for cleaning and preprocessing text data
for the fake news classification project.
"""

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

class TextPreprocessor:
    """A class for preprocessing text data for NLP tasks."""
    
    def __init__(self):
        """Initialize the TextPreprocessor with required NLTK data."""
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except:
            print("Warning: Could not download NLTK data")
        
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """
        Basic text cleaning and normalization.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters and digits (optional)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def advanced_text_processing(self, text, use_stemming=True, remove_stopwords=True):
        """
        Advanced text processing with tokenization, stopword removal, and stemming/lemmatization.
        
        Args:
            text (str): Text to process
            use_stemming (bool): Whether to use stemming (True) or lemmatization (False)
            remove_stopwords (bool): Whether to remove stopwords
            
        Returns:
            str: Processed text
        """
        if pd.isna(text) or text == "":
            return ""
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [token for token in tokens if token.lower() not in self.stop_words]
        
        # Apply stemming or lemmatization
        if use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        else:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Remove very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, title_col='title', text_col='text'):
        """
        Preprocess a dataframe with title and text columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
            title_col (str): Name of title column
            text_col (str): Name of text column
            
        Returns:
            pd.DataFrame: Dataframe with additional processed columns
        """
        df = df.copy()
        
        # Basic cleaning
        df['title_clean'] = df[title_col].apply(self.clean_text)
        df['text_clean'] = df[text_col].apply(self.clean_text)
        
        # Advanced processing
        df['title_processed'] = df['title_clean'].apply(self.advanced_text_processing)
        df['text_processed'] = df['text_clean'].apply(self.advanced_text_processing)
        
        # Combine title and text
        df['combined_text'] = df['title_processed'] + ' ' + df['text_processed']
        
        return df

def get_text_statistics(df, text_col):
    """
    Get basic statistics about text data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_col (str): Name of text column
        
    Returns:
        dict: Dictionary containing text statistics
    """
    df['text_length'] = df[text_col].str.len()
    df['word_count'] = df[text_col].str.split().str.len()
    
    stats = {
        'total_docs': len(df),
        'avg_length': df['text_length'].mean(),
        'median_length': df['text_length'].median(),
        'avg_word_count': df['word_count'].mean(),
        'median_word_count': df['word_count'].median(),
        'min_length': df['text_length'].min(),
        'max_length': df['text_length'].max()
    }
    
    return stats