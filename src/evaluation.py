"""
Evaluation Utilities
Provides metrics, visualization, and robustness testing for AI text detection models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)
import os


def evaluate_model(y_true, y_pred, y_proba=None, model_name="Model"):
    """
    Comprehensive evaluation of a classification model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)
        model_name: Name of the model
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1': f1_score(y_true, y_pred, average='binary')
    }
    
    if y_proba is not None:
        # For binary classification, use probabilities of positive class
        if len(y_proba.shape) > 1:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba
        metrics['auc_roc'] = roc_auc_score(y_true, y_proba_pos)
    
    return metrics


def print_evaluation_report(y_true, y_pred, y_proba=None, model_name="Model"):
    """
    Print detailed evaluation report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)
        model_name: Name of the model
    """
    print(f"\n{'=' * 60}")
    print(f"EVALUATION REPORT: {model_name}")
    print(f"{'=' * 60}\n")
    
    # Basic metrics
    metrics = evaluate_model(y_true, y_pred, y_proba, model_name)
    
    print("Performance Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    if 'auc_roc' in metrics:
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  TN: {cm[0, 0]:4d}  |  FP: {cm[0, 1]:4d}")
    print(f"  FN: {cm[1, 0]:4d}  |  TP: {cm[1, 1]:4d}")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, 
                                target_names=['Human', 'AI-Generated'],
                                digits=4))
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None, return_fig=False):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Path to save the plot
        return_fig: If True, return figure instead of showing
    
    Returns:
        Figure object if return_fig=True, otherwise None
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Human', 'AI-Generated'],
                yticklabels=['Human', 'AI-Generated'], ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    if return_fig:
        return fig
    
    if not save_path:
        plt.show()
    plt.close()
    return None


def plot_roc_curve(y_true, y_proba, model_name="Model", save_path=None, return_fig=False):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_proba: Prediction probabilities
        model_name: Name of the model
        save_path: Path to save the plot
        return_fig: If True, return figure instead of showing
    
    Returns:
        Figure object if return_fig=True, otherwise None
    """
    # For binary classification, use probabilities of positive class
    if len(y_proba.shape) > 1:
        y_proba_pos = y_proba[:, 1]
    else:
        y_proba_pos = y_proba
    
    fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    if return_fig:
        return fig
    
    if not save_path:
        plt.show()
    plt.close()
    return None


def compare_models(results_dict, save_path=None, return_fig=False):
    """
    Compare multiple models and visualize results.
    
    Args:
        results_dict: Dictionary of {model_name: metrics_dict}
        save_path: Path to save the plot
        return_fig: If True, return figure instead of showing
    
    Returns:
        Tuple of (DataFrame, Figure) if return_fig=True, otherwise just DataFrame
    """
    df = pd.DataFrame(results_dict).T
    
    # Plot comparison
    from math import pi
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot of main metrics
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    if 'auc_roc' in df.columns:
        metrics_to_plot.append('auc_roc')
    
    df[metrics_to_plot].plot(kind='bar', ax=axes[0], rot=45,
                            color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    axes[0].set_title('Model Comparison - Main Metrics')
    axes[0].set_ylabel('Score')
    axes[0].set_ylim([0, 1])
    axes[0].legend(loc='lower right', fontsize=8)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_xlabel('Model')
    
    # Radar chart
    categories = metrics_to_plot
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    ax = plt.subplot(1, 2, 2, polar=True)
    colors = ['#1f77b4', '#ff7f0e', '#9467bd']
    
    for idx, (model_name, row) in enumerate(df.iterrows()):
        values = row[metrics_to_plot].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=8)
    ax.set_ylim(0, 1)
    ax.set_title('Performance Radar Chart', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    if return_fig:
        return df, fig
    
    plt.show()
    plt.close()
    return df


def paraphrase_robustness_test(model, original_texts, paraphrased_texts, 
                               labels, model_name="Model"):
    """
    Test model robustness against paraphrased text.
    
    Args:
        model: Trained model with predict method
        original_texts: Original texts
        paraphrased_texts: Paraphrased versions
        labels: True labels
        model_name: Name of the model
        
    Returns:
        Dictionary of robustness metrics
    """
    print(f"\n{'=' * 60}")
    print(f"ROBUSTNESS TEST: {model_name}")
    print(f"{'=' * 60}\n")
    
    # Predictions on original
    original_preds = model.predict(original_texts)
    original_acc = accuracy_score(labels, original_preds)
    
    # Predictions on paraphrased
    paraphrased_preds = model.predict(paraphrased_texts)
    paraphrased_acc = accuracy_score(labels, paraphrased_preds)
    
    # Consistency: same prediction for original and paraphrased
    consistency = np.mean(original_preds == paraphrased_preds)
    
    # Detailed analysis
    print(f"Original Text Accuracy:     {original_acc:.4f}")
    print(f"Paraphrased Text Accuracy:  {paraphrased_acc:.4f}")
    print(f"Prediction Consistency:     {consistency:.4f}")
    print(f"Accuracy Drop:              {(original_acc - paraphrased_acc):.4f}")
    
    results = {
        'original_accuracy': original_acc,
        'paraphrased_accuracy': paraphrased_acc,
        'consistency': consistency,
        'accuracy_drop': original_acc - paraphrased_acc
    }
    
    return results


def analyze_errors(texts, y_true, y_pred, top_n=10):
    """
    Analyze misclassified examples.
    
    Args:
        texts: List of texts
        y_true: True labels
        y_pred: Predicted labels
        top_n: Number of examples to show
        
    Returns:
        DataFrame of misclassified examples
    """
    errors = []
    for i, (text, true_label, pred_label) in enumerate(zip(texts, y_true, y_pred)):
        if true_label != pred_label:
            errors.append({
                'index': i,
                'text': text[:200] + '...' if len(text) > 200 else text,
                'true_label': 'AI' if true_label == 1 else 'Human',
                'predicted_label': 'AI' if pred_label == 1 else 'Human',
                'text_length': len(text)
            })
    
    errors_df = pd.DataFrame(errors)
    
    print(f"\nTotal Errors: {len(errors_df)}")
    print(f"\nFirst {min(top_n, len(errors_df))} misclassified examples:")
    print(errors_df.head(top_n).to_string(index=False))
    
    return errors_df


def get_classification_report(y_true, y_pred, target_names=None, output_dict=False):
    """
    Get classification report as dictionary or DataFrame.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: List of target class names (optional)
        output_dict: If True, return as dictionary; if False, return as DataFrame
        
    Returns:
        Dictionary or DataFrame of classification report
    """
    if target_names is None:
        target_names = ['Human', 'AI-Generated']
    
    report = classification_report(y_true, y_pred, 
                                   target_names=target_names,
                                   output_dict=True)
    
    if output_dict:
        return report
    else:
        return pd.DataFrame(report).T


def save_results(results_dict, output_path):
    """
    Save evaluation results to file.
    
    Args:
        results_dict: Dictionary of results
        output_path: Path to save results
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df = pd.DataFrame([results_dict])
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    print("Evaluation Utilities")
    print("=" * 60)
    print("\nThis module provides comprehensive evaluation tools including:")
    print("  - Standard metrics (accuracy, precision, recall, F1, AUC)")
    print("  - Visualization (confusion matrix, ROC curves)")
    print("  - Model comparison")
    print("  - Robustness testing against paraphrased text")
    print("  - Error analysis")
    print("\nImport and use these functions in your training scripts or notebooks.")
