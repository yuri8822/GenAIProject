"""
Baseline Model: Logistic Regression with Stylometric Features
Uses hand-crafted features to classify AI vs human text.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import TextPreprocessor, split_dataset


class BaselineClassifier:
    """Logistic Regression classifier using stylometric features."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize baseline classifier.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                class_weight='balanced'
            ))
        ])
        self.feature_names = None
    
    def train(self, features_df: pd.DataFrame, labels: list):
        """
        Train the baseline classifier.
        
        Args:
            features_df: DataFrame of stylometric features
            labels: List of labels (0=human, 1=AI)
        """
        self.feature_names = features_df.columns.tolist()
        X = features_df.values
        y = np.array(labels)
        
        print(f"Training baseline model on {len(X)} samples...")
        print(f"Features: {len(self.feature_names)}")
        print(f"Class distribution: {np.bincount(y)}")
        
        self.pipeline.fit(X, y)
        
        # Feature importance (coefficients)
        coefs = self.pipeline.named_steps['classifier'].coef_[0]
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefs,
            'abs_coefficient': np.abs(coefs)
        }).sort_values('abs_coefficient', ascending=False)
        
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
        
        return feature_importance
    
    def predict(self, features_df: pd.DataFrame):
        """
        Make predictions on new data.
        
        Args:
            features_df: DataFrame of stylometric features
            
        Returns:
            Array of predictions
        """
        X = features_df.values
        return self.pipeline.predict(X)
    
    def predict_proba(self, features_df: pd.DataFrame):
        """
        Get prediction probabilities.
        
        Args:
            features_df: DataFrame of stylometric features
            
        Returns:
            Array of probabilities
        """
        X = features_df.values
        return self.pipeline.predict_proba(X)
    
    def save_model(self, filepath: str):
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'feature_names': self.feature_names,
                'random_state': self.random_state
            }, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to saved model
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.pipeline = data['pipeline']
            self.feature_names = data['feature_names']
            self.random_state = data['random_state']
        print(f"Model loaded from {filepath}")


def train_baseline_model(data_dir: str, output_dir: str):
    """
    Complete training pipeline for baseline model.
    
    Args:
        data_dir: Directory containing train.csv, val.csv, test.csv
        output_dir: Directory to save model and results
    """
    print("=" * 60)
    print("BASELINE MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    print(f"   Train size: {len(train_df)}")
    print(f"   Test size: {len(test_df)}")
    
    # Preprocess and extract features
    print("\n2. Preprocessing and extracting features...")
    preprocessor = TextPreprocessor()
    
    train_texts, train_labels, train_features = preprocessor.preprocess_dataset(
        train_df['text'].tolist(), 
        train_df['label'].tolist(), 
        extract_features=True
    )
    
    test_texts, test_labels, test_features = preprocessor.preprocess_dataset(
        test_df['text'].tolist(), 
        test_df['label'].tolist(), 
        extract_features=True
    )
    
    # Train model
    print("\n3. Training baseline classifier...")
    classifier = BaselineClassifier()
    feature_importance = classifier.train(train_features, train_labels)
    
    # Save model
    model_path = os.path.join(output_dir, 'baseline_model.pkl')
    classifier.save_model(model_path)
    
    # Save feature importance
    importance_path = os.path.join(output_dir, 'feature_importance.csv')
    feature_importance.to_csv(importance_path, index=False)
    print(f"\nFeature importance saved to {importance_path}")
    
    # Evaluate on test set
    print("\n4. Evaluating on test set...")
    test_pred = classifier.predict(test_features)
    test_proba = classifier.predict_proba(test_features)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(test_labels, test_pred)
    precision = precision_score(test_labels, test_pred)
    recall = recall_score(test_labels, test_pred)
    f1 = f1_score(test_labels, test_pred)
    
    print(f"\nTest Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    output_dir = os.path.join(project_root, 'models')
    
    train_baseline_model(data_dir, output_dir)
