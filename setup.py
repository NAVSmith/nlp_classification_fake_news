#!/usr/bin/env python3
"""
Setup script for the Fake News Classification Project

This script helps set up the environment and download necessary data.
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages from requirements.txt"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully!")
    except subprocess.CalledProcessError:
        print("✗ Error installing requirements. Please install manually.")
        return False
    return True

def download_nltk_data():
    """Download necessary NLTK data"""
    print("Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("✓ NLTK data downloaded successfully!")
    except Exception as e:
        print(f"✗ Error downloading NLTK data: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories"""
    dirs = ['models', 'results', 'dataset', 'notebooks', 'src']
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✓ Created directory: {dir_name}")
        else:
            print(f"✓ Directory already exists: {dir_name}")

def main():
    """Main setup function"""
    print("=" * 50)
    print("Fake News Classification Project Setup")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("Setup incomplete. Please install requirements manually.")
        return
    
    # Download NLTK data
    download_nltk_data()
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Place your data.csv file in the dataset/ folder")
    print("2. Place your validation_data.csv file in the dataset/ folder")
    print("3. Open the notebooks in the notebooks/ folder")
    print("4. Run the notebooks in order:")
    print("   - 01_data_exploration_preprocessing.ipynb")
    print("   - 02_model_training_evaluation.ipynb")

if __name__ == "__main__":
    main()