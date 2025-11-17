"""
Data Preprocessing Module
Handles text cleaning, tokenization, and feature extraction for AI text detection.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import textstat
from lexicalrichness import LexicalRichness


class TextPreprocessor:
    """Handles text preprocessing and feature extraction."""
    
    def __init__(self):
        """Initialize preprocessor and download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and normalizing whitespace.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_stylometric_features(self, text: str) -> Dict[str, float]:
        """
        Extract stylometric features for text classification.
        
        Note: Absolute length features (char_count, word_count, sent_count) are excluded
        to prevent bias based on text length. Only relative stylometric features are used.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary of stylometric features
        """
        features = {}
        
        # Tokenize for feature calculation
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        word_count = len(words)
        sent_count = len(sentences)
        char_count = len(text)
        
        # Average lengths (relative features - keep these)
        if word_count > 0:
            features['avg_word_length'] = char_count / word_count
        else:
            features['avg_word_length'] = 0
            
        if sent_count > 0:
            features['avg_sent_length'] = word_count / sent_count
        else:
            features['avg_sent_length'] = 0
        
        # Readability scores
        try:
            features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
            features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
            features['gunning_fog'] = textstat.gunning_fog(text)
            features['smog_index'] = textstat.smog_index(text)
            features['automated_readability_index'] = textstat.automated_readability_index(text)
        except:
            features['flesch_reading_ease'] = 0
            features['flesch_kincaid_grade'] = 0
            features['gunning_fog'] = 0
            features['smog_index'] = 0
            features['automated_readability_index'] = 0
        
        # Lexical diversity
        try:
            lex = LexicalRichness(text)
            features['ttr'] = lex.ttr  # Type-token ratio
            features['mtld'] = lex.mtld()  # Measure of textual lexical diversity
        except:
            features['ttr'] = 0
            features['mtld'] = 0
        
        # Punctuation density
        punctuation_count = sum([1 for char in text if char in '.,;:!?'])
        features['punctuation_density'] = punctuation_count / len(text) if len(text) > 0 else 0
        
        return features
    
    def preprocess_dataset(self, 
                          texts: List[str], 
                          labels: List[int],
                          extract_features: bool = False) -> Tuple:
        """
        Preprocess a dataset of texts.
        
        Args:
            texts: List of text strings
            labels: List of labels (0 for human, 1 for AI)
            extract_features: Whether to extract stylometric features
            
        Returns:
            Tuple of (cleaned_texts, labels, features_df) if extract_features=True
            Tuple of (cleaned_texts, labels) otherwise
        """
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        if extract_features:
            features_list = [self.extract_stylometric_features(text) for text in cleaned_texts]
            features_df = pd.DataFrame(features_list)
            return cleaned_texts, labels, features_df
        
        return cleaned_texts, labels


def load_hc3_dataset(filepath: str) -> Tuple[List[str], List[int]]:
    """
    Load HC3 dataset from file.
    
    Args:
        filepath: Path to HC3 dataset file
        
    Returns:
        Tuple of (texts, labels)
    """
    # Placeholder - implement based on actual HC3 dataset format
    # Expected format: JSON or CSV with 'text' and 'label' columns
    df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_json(filepath)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return texts, labels


def load_kaggle_dataset(filepath: str) -> Tuple[List[str], List[int]]:
    """
    Load Kaggle AI vs Human dataset from file.
    
    Args:
        filepath: Path to Kaggle dataset file
        
    Returns:
        Tuple of (texts, labels)
    """
    # Placeholder - implement based on actual Kaggle dataset format
    df = pd.read_csv(filepath)
    texts = df['text'].tolist()
    labels = df['generated'].tolist()  # Adjust column name as needed
    return texts, labels


def split_dataset(texts: List[str], 
                 labels: List[int], 
                 features_df: pd.DataFrame = None,
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 random_state: int = 42) -> Dict:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        texts: List of text strings
        labels: List of labels
        features_df: DataFrame of features (optional)
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
        
    Returns:
        Dictionary containing split datasets
    """
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test
    if features_df is not None:
        train_val_texts, test_texts, train_val_labels, test_labels, train_val_features, test_features = \
            train_test_split(texts, labels, features_df, test_size=test_size, random_state=random_state, stratify=labels)
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train_texts, val_texts, train_labels, val_labels, train_features, val_features = \
            train_test_split(train_val_texts, train_val_labels, train_val_features, 
                           test_size=val_ratio, random_state=random_state, stratify=train_val_labels)
        
        return {
            'train': {'texts': train_texts, 'labels': train_labels, 'features': train_features},
            'val': {'texts': val_texts, 'labels': val_labels, 'features': val_features},
            'test': {'texts': test_texts, 'labels': test_labels, 'features': test_features}
        }
    else:
        train_val_texts, test_texts, train_val_labels, test_labels = \
            train_test_split(texts, labels, test_size=test_size, random_state=random_state, stratify=labels)
        
        val_ratio = val_size / (1 - test_size)
        train_texts, val_texts, train_labels, val_labels = \
            train_test_split(train_val_texts, train_val_labels, 
                           test_size=val_ratio, random_state=random_state, stratify=train_val_labels)
        
        return {
            'train': {'texts': train_texts, 'labels': train_labels},
            'val': {'texts': val_texts, 'labels': val_labels},
            'test': {'texts': test_texts, 'labels': test_labels}
        }


if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor()
    
    # Test text
    sample_text = "This is a sample text for testing preprocessing capabilities. It has multiple sentences!"
    
    # Clean text
    cleaned = preprocessor.clean_text(sample_text)
    print(f"Cleaned text: {cleaned}\n")
    
    # Extract features
    features = preprocessor.extract_stylometric_features(sample_text)
    print("Stylometric features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
