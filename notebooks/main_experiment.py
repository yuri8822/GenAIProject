"""
AI Text Detection - Main Experiment Notebook
This notebook provides a complete pipeline for training and evaluating AI text detection models.
"""

# Cell 1: Setup and Imports
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Add src to path
sys.path.append('../src')

print("Environment setup complete!")
print(f"Working directory: {os.getcwd()}")

# Cell 2: Import Custom Modules
from preprocessing import TextPreprocessor, split_dataset
from baseline_model import BaselineClassifier
from bert_classifier import BERTClassifier
from roberta_classifier import RoBERTaClassifier
from evaluation import (
    evaluate_model, 
    print_evaluation_report,
    plot_confusion_matrix,
    plot_roc_curve,
    compare_models,
    paraphrase_robustness_test
)

print("Custom modules imported successfully!")

# Cell 3: Load Dataset
"""
DATASET LOADING
===============
Replace this cell with code to load your actual dataset.
Expected format: texts (list of strings), labels (list of 0/1)

Example for HC3 dataset:
    df = pd.read_json('../data/hc3_english.json')
    texts = df['text'].tolist()
    labels = df['label'].tolist()  # 0=human, 1=AI

Example for Kaggle dataset:
    df = pd.read_csv('../data/ai_vs_human.csv')
    texts = df['text'].tolist()
    labels = df['generated'].map({'human': 0, 'ai': 1}).tolist()
"""

# Placeholder - create sample data for demonstration
print("Creating sample dataset for demonstration...")
sample_size = 1000

# Generate synthetic data (replace with actual data loading)
np.random.seed(42)
texts = [f"Sample text number {i} with some content." for i in range(sample_size)]
labels = np.random.randint(0, 2, sample_size).tolist()

print(f"Dataset loaded: {len(texts)} samples")
print(f"Label distribution: {np.bincount(labels)}")

# Cell 4: Data Preprocessing
print("Preprocessing data...")
preprocessor = TextPreprocessor()

# Clean texts and extract features
cleaned_texts, labels, features_df = preprocessor.preprocess_dataset(
    texts, labels, extract_features=True
)

print(f"Preprocessing complete!")
print(f"Feature shape: {features_df.shape}")
print(f"\nSample features:")
print(features_df.head())

# Cell 5: Train-Validation-Test Split
print("Splitting dataset...")
data_splits = split_dataset(
    cleaned_texts, 
    labels, 
    features_df,
    test_size=0.2,
    val_size=0.1,
    random_state=42
)

print(f"Train size: {len(data_splits['train']['texts'])}")
print(f"Val size: {len(data_splits['val']['texts'])}")
print(f"Test size: {len(data_splits['test']['texts'])}")

# Cell 6: Train Baseline Model
print("\n" + "="*60)
print("TRAINING BASELINE MODEL (Logistic Regression)")
print("="*60 + "\n")

baseline_model = BaselineClassifier(random_state=42)
feature_importance = baseline_model.train(
    data_splits['train']['features'],
    data_splits['train']['labels']
)

# Save model
baseline_model.save_model('../models/baseline_model.pkl')

# Cell 7: Evaluate Baseline Model
print("\nEvaluating baseline model on test set...")
baseline_test_pred = baseline_model.predict(data_splits['test']['features'])
baseline_test_proba = baseline_model.predict_proba(data_splits['test']['features'])

baseline_metrics = print_evaluation_report(
    data_splits['test']['labels'],
    baseline_test_pred,
    baseline_test_proba,
    "Baseline (Logistic Regression)"
)

# Cell 8: Visualize Baseline Results
plot_confusion_matrix(
    data_splits['test']['labels'],
    baseline_test_pred,
    "Baseline Model",
    save_path='../results/baseline_confusion_matrix.png'
)

plot_roc_curve(
    data_splits['test']['labels'],
    baseline_test_proba,
    "Baseline Model",
    save_path='../results/baseline_roc_curve.png'
)

# Cell 9: Train BERT Model
print("\n" + "="*60)
print("TRAINING BERT MODEL")
print("="*60 + "\n")

# Note: This may take significant time and GPU resources
bert_model = BERTClassifier(
    model_name='bert-base-uncased',
    max_length=256  # Reduced for faster training
)

# Train (reduce epochs for quick testing)
bert_model.train(
    data_splits['train']['texts'],
    data_splits['train']['labels'],
    data_splits['val']['texts'],
    data_splits['val']['labels'],
    batch_size=16,
    epochs=2,  # Increase to 3-5 for better results
    learning_rate=2e-5
)

# Save model
bert_model.save_model('../models/bert_model')

# Cell 10: Evaluate BERT Model
print("\nEvaluating BERT model on test set...")
bert_test_pred = bert_model.predict(data_splits['test']['texts'], batch_size=16)
bert_test_proba = bert_model.predict_proba(data_splits['test']['texts'], batch_size=16)

bert_metrics = print_evaluation_report(
    data_splits['test']['labels'],
    bert_test_pred,
    bert_test_proba,
    "BERT"
)

# Cell 11: Visualize BERT Results
plot_confusion_matrix(
    data_splits['test']['labels'],
    bert_test_pred,
    "BERT Model",
    save_path='../results/bert_confusion_matrix.png'
)

plot_roc_curve(
    data_splits['test']['labels'],
    bert_test_proba,
    "BERT Model",
    save_path='../results/bert_roc_curve.png'
)

# Cell 12: Train RoBERTa Model
print("\n" + "="*60)
print("TRAINING ROBERTA MODEL")
print("="*60 + "\n")

roberta_model = RoBERTaClassifier(
    model_name='roberta-base',
    max_length=256
)

roberta_model.train(
    data_splits['train']['texts'],
    data_splits['train']['labels'],
    data_splits['val']['texts'],
    data_splits['val']['labels'],
    batch_size=16,
    epochs=2,  # Increase to 3-5 for better results
    learning_rate=2e-5
)

# Save model
roberta_model.save_model('../models/roberta_model')

# Cell 13: Evaluate RoBERTa Model
print("\nEvaluating RoBERTa model on test set...")
roberta_test_pred = roberta_model.predict(data_splits['test']['texts'], batch_size=16)
roberta_test_proba = roberta_model.predict_proba(data_splits['test']['texts'], batch_size=16)

roberta_metrics = print_evaluation_report(
    data_splits['test']['labels'],
    roberta_test_pred,
    roberta_test_proba,
    "RoBERTa"
)

# Cell 14: Visualize RoBERTa Results
plot_confusion_matrix(
    data_splits['test']['labels'],
    roberta_test_pred,
    "RoBERTa Model",
    save_path='../results/roberta_confusion_matrix.png'
)

plot_roc_curve(
    data_splits['test']['labels'],
    roberta_test_proba,
    "RoBERTa Model",
    save_path='../results/roberta_roc_curve.png'
)

# Cell 15: Compare All Models
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60 + "\n")

results = {
    'Baseline': baseline_metrics,
    'BERT': bert_metrics,
    'RoBERTa': roberta_metrics
}

comparison_df = compare_models(results, save_path='../results/model_comparison.png')
print("\nModel Comparison Summary:")
print(comparison_df)

# Cell 16: Robustness Testing (Optional)
"""
ROBUSTNESS TESTING
==================
To test robustness, you need paraphrased versions of AI-generated texts.
You can use tools like:
- ChatGPT/GPT-4 API
- Pegasus paraphraser
- Back-translation

Example:
    paraphrased_texts = paraphrase_texts(ai_generated_texts)
    robustness_results = paraphrase_robustness_test(
        bert_model,
        original_texts,
        paraphrased_texts,
        labels,
        "BERT"
    )
"""

print("For robustness testing, prepare paraphrased versions of your test set.")
print("Refer to Cell 16 for implementation guidance.")

# Cell 17: Save Final Results
results_summary = pd.DataFrame(results).T
results_summary.to_csv('../results/final_results.csv')

print("\n" + "="*60)
print("EXPERIMENT COMPLETE!")
print("="*60)
print("\nAll models trained and evaluated.")
print(f"Results saved to: ../results/")
print("\nNext steps:")
print("1. Review model performance metrics")
print("2. Analyze error cases")
print("3. Test robustness with paraphrased text")
print("4. Fine-tune hyperparameters if needed")
