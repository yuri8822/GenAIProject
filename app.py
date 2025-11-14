"""
Streamlit UI for AI Text Detection
Simple interface to view model statistics and classify text.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.baseline_model import BaselineClassifier
from src.bert_classifier import BERTClassifier
from src.roberta_classifier import RoBERTaClassifier
from src.preprocessing import TextPreprocessor

# Set page config
st.set_page_config(
    page_title="AI Text Detection",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.baseline_model = None
    st.session_state.bert_model = None
    st.session_state.roberta_model = None
    st.session_state.preprocessor = TextPreprocessor()


@st.cache_resource
def load_baseline_model():
    """Load baseline model from file."""
    model_path = 'models/baseline_model.pkl'
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            
            # Check if it's a dictionary (saved format) or already a model object
            if isinstance(data, dict):
                # Reconstruct the model object
                model = BaselineClassifier(random_state=data.get('random_state', 42))
                model.pipeline = data['pipeline']
                model.feature_names = data['feature_names']
                return model
            else:
                # Already a model object
                return data
        except Exception as e:
            st.error(f"Error loading baseline model: {e}")
            return None
    return None


@st.cache_resource
def load_bert_model():
    """Load BERT model from file."""
    model_path = 'models/bert_model.pt'
    if os.path.exists(model_path):
        try:
            model = BERTClassifier()
            model.model.load_state_dict(torch.load(model_path, map_location=model.device))
            model.model.eval()
            return model
        except Exception as e:
            st.warning(f"Could not load BERT model: {e}")
            return None
    return None


@st.cache_resource
def load_roberta_model():
    """Load RoBERTa model from file."""
    model_path = 'models/roberta_model.pt'
    if os.path.exists(model_path):
        try:
            model = RoBERTaClassifier()
            model.model.load_state_dict(torch.load(model_path, map_location=model.device))
            model.model.eval()
            return model
        except Exception as e:
            st.warning(f"Could not load RoBERTa model: {e}")
            return None
    return None


def load_feature_importance():
    """Load feature importance data."""
    fi_path = 'models/feature_importance.csv'
    if os.path.exists(fi_path):
        return pd.read_csv(fi_path)
    return None


def classify_text_baseline(model, preprocessor, text):
    """Classify text using baseline model."""
    if model is None:
        return None, None
    
    try:
        cleaned_text = preprocessor.clean_text(text)
        features = preprocessor.extract_stylometric_features(cleaned_text)
        features_df = pd.DataFrame([features])
        
        # Ensure all expected features are present
        if hasattr(model, 'feature_names') and model.feature_names:
            for feat in model.feature_names:
                if feat not in features_df.columns:
                    features_df[feat] = 0
            features_df = features_df[model.feature_names]
        
        prediction = model.predict(features_df)[0]
        proba = model.predict_proba(features_df)[0]
        
        return prediction, proba
    except Exception as e:
        st.error(f"Error in baseline classification: {e}")
        return None, None


def classify_text_transformer(model, text):
    """Classify text using transformer model (BERT/RoBERTa)."""
    if model is None:
        return None, None
    
    try:
        prediction, proba = model.predict_single(text)
        return prediction, proba
    except Exception as e:
        st.error(f"Error in transformer classification: {e}")
        return None, None


def main():
    st.title("🤖 AI Text Detection System")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Model Statistics", "Text Classification", "Train Models"])
    
    # Load models
    baseline_model = load_baseline_model()
    bert_model = load_bert_model()
    roberta_model = load_roberta_model()
    
    # Model availability status
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Status")
    st.sidebar.write("✅ Baseline" if baseline_model else "❌ Baseline")
    st.sidebar.write("✅ BERT" if bert_model else "❌ BERT")
    st.sidebar.write("✅ RoBERTa" if roberta_model else "❌ RoBERTa")
    
    if page == "Model Statistics":
        show_statistics(baseline_model, bert_model, roberta_model)
    elif page == "Text Classification":
        show_classification(baseline_model, bert_model, roberta_model)
    else:
        show_training()


def show_statistics(baseline_model, bert_model, roberta_model):
    """Display model statistics page."""
    st.header("📊 Model Statistics")
    
    # Check for results files
    results_files = []
    results_dir = 'results'
    if os.path.exists(results_dir):
        results_files = [f for f in os.listdir(results_dir) if f.endswith('_metrics.csv')]
    
    if not results_files:
        st.warning("No model statistics found. Train models first to see statistics.")
        st.info("Expected files in `results/` directory: model evaluation metrics CSVs")
    else:
        st.success(f"Found {len(results_files)} results file(s)")
        
        # Display metrics summary
        st.subheader("📈 Performance Metrics")
        metrics_data = {}
        for file in results_files:
            model_name = file.replace('_metrics.csv', '')
            df = pd.read_csv(os.path.join(results_dir, file))
            metrics_data[model_name] = df.iloc[0].to_dict()
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data).T
            st.dataframe(metrics_df, width='stretch')
            
            # Display visualizations if they exist
            st.markdown("---")
            st.subheader("� Confusion Matrices & ROC Curves")
            
            available_models = []
            for model_name in ['baseline', 'bert', 'roberta']:
                cm_path = f'results/{model_name}_confusion_matrix.png'
                roc_path = f'results/{model_name}_roc_curve.png'
                if os.path.exists(cm_path) and os.path.exists(roc_path):
                    available_models.append(model_name)
            
            if available_models:
                for model_name in available_models:
                    st.markdown(f"### {model_name.upper()}")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        cm_path = f'results/{model_name}_confusion_matrix.png'
                        if os.path.exists(cm_path):
                            st.image(cm_path, caption=f'{model_name.title()} Confusion Matrix')
                    
                    with col2:
                        roc_path = f'results/{model_name}_roc_curve.png'
                        if os.path.exists(roc_path):
                            st.image(roc_path, caption=f'{model_name.title()} ROC Curve')
            
            # Model comparison visualization
            if os.path.exists('results/model_comparison.png'):
                st.markdown("---")
                st.subheader("🔄 Model Comparison")
                st.image('results/model_comparison.png', caption='Model Performance Comparison')
    
    # Feature importance for baseline
    st.markdown("---")
    st.subheader("🎯 Feature Importance (Baseline Model)")
    
    feature_importance = load_feature_importance()
    if feature_importance is not None:
        st.dataframe(feature_importance.head(15), width='stretch')
        
        # Visualize top features
        st.bar_chart(
            feature_importance.head(10).set_index('feature')['abs_coefficient']
        )
    else:
        st.info("Feature importance data not found. Train the baseline model to generate this data.")
    
    # Model information
    st.markdown("---")
    st.subheader("ℹ️ Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Baseline Model**")
        if baseline_model:
            st.write("- Type: Logistic Regression")
            st.write("- Features: Stylometric")
            if hasattr(baseline_model, 'feature_names'):
                st.write(f"- # Features: {len(baseline_model.feature_names)}")
        else:
            st.write("Not loaded")
    
    with col2:
        st.markdown("**BERT Model**")
        if bert_model:
            st.write("- Type: BERT")
            st.write("- Base: bert-base-uncased")
            st.write(f"- Device: {bert_model.device}")
        else:
            st.write("Not loaded")
    
    with col3:
        st.markdown("**RoBERTa Model**")
        if roberta_model:
            st.write("- Type: RoBERTa")
            st.write("- Base: roberta-base")
            st.write(f"- Device: {roberta_model.device}")
        else:
            st.write("Not loaded")


def show_classification(baseline_model, bert_model, roberta_model):
    """Display text classification page."""
    st.header("✍️ Text Classification")
    st.write("Enter text below to classify it as AI-generated or Human-written.")
    
    # Text input
    text_input = st.text_area(
        "Enter text to classify:",
        height=200,
        placeholder="Type or paste your text here..."
    )
    
    # Model selection
    st.subheader("Select Models to Use")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_baseline = st.checkbox("Baseline", value=baseline_model is not None, 
                                   disabled=baseline_model is None)
    with col2:
        use_bert = st.checkbox("BERT", value=bert_model is not None, 
                              disabled=bert_model is None)
    with col3:
        use_roberta = st.checkbox("RoBERTa", value=roberta_model is not None, 
                                 disabled=roberta_model is None)
    
    # Classify button
    if st.button("🔍 Classify", type="primary"):
        if not text_input.strip():
            st.warning("Please enter some text to classify.")
            return
        
        if not (use_baseline or use_bert or use_roberta):
            st.warning("Please select at least one model.")
            return
        
        st.markdown("---")
        st.subheader("📊 Classification Results")
        
        preprocessor = st.session_state.preprocessor
        
        # Create columns for results
        results = []
        
        # Baseline classification
        if use_baseline and baseline_model:
            with st.spinner("Running Baseline model..."):
                pred, proba = classify_text_baseline(baseline_model, preprocessor, text_input)
                if pred is not None:
                    results.append({
                        'Model': 'Baseline',
                        'Prediction': 'AI-Generated' if pred == 1 else 'Human-Written',
                        'Confidence': f"{max(proba) * 100:.2f}%",
                        'Human Prob': f"{proba[0] * 100:.2f}%",
                        'AI Prob': f"{proba[1] * 100:.2f}%"
                    })
        
        # BERT classification
        if use_bert and bert_model:
            with st.spinner("Running BERT model..."):
                pred, proba = classify_text_transformer(bert_model, text_input)
                if pred is not None:
                    results.append({
                        'Model': 'BERT',
                        'Prediction': 'AI-Generated' if pred == 1 else 'Human-Written',
                        'Confidence': f"{max(proba) * 100:.2f}%",
                        'Human Prob': f"{proba[0] * 100:.2f}%",
                        'AI Prob': f"{proba[1] * 100:.2f}%"
                    })
        
        # RoBERTa classification
        if use_roberta and roberta_model:
            with st.spinner("Running RoBERTa model..."):
                pred, proba = classify_text_transformer(roberta_model, text_input)
                if pred is not None:
                    results.append({
                        'Model': 'RoBERTa',
                        'Prediction': 'AI-Generated' if pred == 1 else 'Human-Written',
                        'Confidence': f"{max(proba) * 100:.2f}%",
                        'Human Prob': f"{proba[0] * 100:.2f}%",
                        'AI Prob': f"{proba[1] * 100:.2f}%"
                    })
        
        # Display results
        if results:
            results_df = pd.DataFrame(results)
            
            # Display with styling
            for result in results:
                with st.container():
                    col1, col2, col3 = st.columns([2, 3, 3])
                    
                    with col1:
                        st.markdown(f"**{result['Model']}**")
                    
                    with col2:
                        if result['Prediction'] == 'AI-Generated':
                            st.error(f"🤖 {result['Prediction']}")
                        else:
                            st.success(f"👤 {result['Prediction']}")
                    
                    with col3:
                        st.info(f"Confidence: {result['Confidence']}")
                    
                    # Progress bars for probabilities
                    st.progress(float(result['Human Prob'].strip('%')) / 100, 
                               text=f"Human: {result['Human Prob']}")
                    st.progress(float(result['AI Prob'].strip('%')) / 100, 
                               text=f"AI: {result['AI Prob']}")
                    
                    st.markdown("---")
            
            # Summary table
            st.subheader("Summary Table")
            st.dataframe(results_df, width='stretch')
        else:
            st.error("No results to display. Check if models are loaded correctly.")
    
    # Example texts
    st.markdown("---")
    st.subheader("📝 Example Texts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Human-Written Example:**")
        if st.button("Load Human Example"):
            human_example = """I've always found it fascinating how the city comes alive at night. The streets, 
            which during the day are filled with the hustle and bustle of commuters and shoppers, 
            transform into something entirely different. There's a certain magic in the way the neon 
            lights reflect off the wet pavement after rain, creating these beautiful, shimmering pools 
            of color. Sometimes I just walk around aimlessly, taking it all in."""
            st.session_state.example_text = human_example
            st.rerun()
    
    with col2:
        st.markdown("**AI-Generated Example:**")
        if st.button("Load AI Example"):
            ai_example = """Artificial intelligence has revolutionized numerous industries by providing 
            efficient solutions to complex problems. Machine learning algorithms can process vast amounts 
            of data and identify patterns that would be impossible for humans to detect manually. 
            This technology has applications in healthcare, finance, transportation, and many other 
            sectors. As AI continues to advance, it promises to bring even more transformative changes 
            to our daily lives and work environments."""
            st.session_state.example_text = ai_example
            st.rerun()


def show_training():
    """Display model training page."""
    st.header("🎓 Train Models")
    st.write("Train models on the HC3 dataset. This page allows you to train all three models.")
    
    # Check for dataset
    train_file = 'data/train.csv'
    val_file = 'data/val.csv'
    
    dataset_exists = os.path.exists(train_file) and os.path.exists(val_file)
    
    if not dataset_exists:
        st.error("⚠️ Training and validation datasets not found!")
        st.info("""
        Required files:
        - `data/train.csv`
        - `data/val.csv`
        
        These should have been created from the HC3 dataset preparation script.
        """)
        return
    
    # Show dataset info
    try:
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        
        st.success(f"✅ Dataset found: {len(train_df)} training samples, {len(val_df)} validation samples")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Samples", len(train_df))
            if 'label' in train_df.columns:
                train_dist = train_df['label'].value_counts()
                st.write(f"Human: {train_dist.get(0, 0)}, AI: {train_dist.get(1, 0)}")
        
        with col2:
            st.metric("Validation Samples", len(val_df))
            if 'label' in val_df.columns:
                val_dist = val_df['label'].value_counts()
                st.write(f"Human: {val_dist.get(0, 0)}, AI: {val_dist.get(1, 0)}")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return
    
    st.markdown("---")
    
    # Training options
    st.subheader("⚙️ Training Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        train_baseline = st.checkbox("Train Baseline", value=True)
    with col2:
        train_bert = st.checkbox("Train BERT", value=False)
    with col3:
        train_roberta = st.checkbox("Train RoBERTa", value=False)
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transformer Settings")
            st.info("ℹ️ These settings apply only to BERT and RoBERTa models, not the Baseline model.")
            epochs = st.number_input("Epochs", min_value=1, max_value=10, value=3)
            batch_size = st.number_input("Batch Size", min_value=4, max_value=64, value=16)
            learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-3, 
                                           value=2e-5, format="%.6f")
        
        with col2:
            st.subheader("Output Settings")
            save_models = st.checkbox("Save trained models", value=True)
            save_metrics = st.checkbox("Save evaluation metrics", value=True)
    
    # Train button
    st.markdown("---")
    
    if not (train_baseline or train_bert or train_roberta):
        st.warning("Please select at least one model to train.")
        return
    
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        train_models(
            train_baseline=train_baseline,
            train_bert=train_bert,
            train_roberta=train_roberta,
            train_file=train_file,
            val_file=val_file,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            save_models=save_models,
            save_metrics=save_metrics
        )


def train_models(train_baseline, train_bert, train_roberta, train_file, val_file,
                epochs, batch_size, learning_rate, save_models, save_metrics):
    """Execute model training."""
    
    # Load data
    st.info("📂 Loading dataset...")
    try:
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        
        train_texts = train_df['text'].tolist()
        train_labels = train_df['label'].tolist()
        val_texts = val_df['text'].tolist()
        val_labels = val_df['label'].tolist()
        
        st.success(f"✅ Loaded {len(train_texts)} training and {len(val_texts)} validation samples")
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return
    
    preprocessor = TextPreprocessor()
    results = {}
    
    # Train Baseline Model
    if train_baseline:
        st.markdown("---")
        st.subheader("🔵 Training Baseline Model")
        progress_bar = st.progress(0, text="Extracting features...")
        
        try:
            # Extract features
            progress_bar.progress(25, text="Extracting training features...")
            _, _, train_features = preprocessor.preprocess_dataset(
                train_texts, train_labels, extract_features=True
            )
            
            progress_bar.progress(50, text="Extracting validation features...")
            _, _, val_features = preprocessor.preprocess_dataset(
                val_texts, val_labels, extract_features=True
            )
            
            # Train model
            progress_bar.progress(75, text="Training model...")
            model = BaselineClassifier()
            feature_importance = model.train(train_features, train_labels)
            
            # Evaluate
            progress_bar.progress(90, text="Evaluating...")
            val_pred = model.predict(val_features)
            val_proba = model.predict_proba(val_features)
            
            from src.evaluation import (
                evaluate_model, 
                plot_confusion_matrix, 
                plot_roc_curve,
                get_classification_report
            )
            
            metrics = evaluate_model(val_labels, val_pred, val_proba, "Baseline")
            
            progress_bar.progress(100, text="Complete!")
            
            st.success(f"✅ Baseline trained! Validation Accuracy: {metrics['accuracy']:.4f}")
            
            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            col2.metric("Precision", f"{metrics['precision']:.4f}")
            col3.metric("Recall", f"{metrics['recall']:.4f}")
            col4.metric("F1 Score", f"{metrics['f1']:.4f}")
            if 'auc_roc' in metrics:
                col5.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")
            
            # Visualizations using evaluation.py functions
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Confusion Matrix")
                fig_cm = plot_confusion_matrix(val_labels, val_pred, "Baseline", return_fig=True)
                st.pyplot(fig_cm)
                import matplotlib.pyplot as plt
                plt.close(fig_cm)
            
            with col2:
                st.subheader("ROC Curve")
                fig_roc = plot_roc_curve(val_labels, val_proba, "Baseline", return_fig=True)
                st.pyplot(fig_roc)
                plt.close(fig_roc)
            
            # Classification Report
            with st.expander("📋 Detailed Classification Report"):
                report_df = get_classification_report(val_labels, val_pred, 
                                                     target_names=['Human', 'AI-Generated'])
                st.dataframe(report_df, width='stretch')
            
            results['baseline'] = metrics
            
            # Save model
            if save_models:
                os.makedirs('models', exist_ok=True)
                model.save_model('models/baseline_model.pkl')
                feature_importance.to_csv('models/feature_importance.csv', index=False)
                st.info("💾 Model saved to `models/baseline_model.pkl`")
            
            # Save metrics and visualizations using evaluation.py
            if save_metrics:
                os.makedirs('results', exist_ok=True)
                pd.DataFrame([metrics]).to_csv('results/baseline_metrics.csv', index=False)
                
                # Save classification report
                report_df.to_csv('results/baseline_classification_report.csv')
                
                # Save visualizations using evaluation.py functions
                plot_confusion_matrix(val_labels, val_pred, "Baseline", 
                                    save_path='results/baseline_confusion_matrix.png')
                plot_roc_curve(val_labels, val_proba, "Baseline", 
                             save_path='results/baseline_roc_curve.png')
                
                st.info("📊 Metrics and visualizations saved to `results/`")
                
        except Exception as e:
            st.error(f"❌ Baseline training failed: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Train BERT Model
    if train_bert:
        st.markdown("---")
        st.subheader("🟢 Training BERT Model")
        progress_bar = st.progress(0, text="Initializing BERT...")
        
        try:
            progress_bar.progress(10, text="Loading BERT model...")
            model = BERTClassifier()
            
            progress_bar.progress(20, text="Starting training...")
            st.info(f"Training for {epochs} epochs with batch size {batch_size}")
            
            # Train with progress updates
            model.train(
                train_texts, train_labels,
                val_texts, val_labels,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate
            )
            
            progress_bar.progress(90, text="Evaluating...")
            
            # Evaluate
            val_pred = model.predict(val_texts, batch_size=batch_size)
            val_proba = model.predict_proba(val_texts, batch_size=batch_size)
            
            from src.evaluation import (
                evaluate_model, 
                plot_confusion_matrix, 
                plot_roc_curve,
                get_classification_report
            )
            import matplotlib.pyplot as plt
            
            metrics = evaluate_model(val_labels, val_pred, val_proba, "BERT")
            
            progress_bar.progress(100, text="Complete!")
            
            st.success(f"✅ BERT trained! Validation Accuracy: {metrics['accuracy']:.4f}")
            
            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            col2.metric("Precision", f"{metrics['precision']:.4f}")
            col3.metric("Recall", f"{metrics['recall']:.4f}")
            col4.metric("F1 Score", f"{metrics['f1']:.4f}")
            if 'auc_roc' in metrics:
                col5.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")
            
            # Visualizations using evaluation.py functions
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Confusion Matrix")
                fig_cm = plot_confusion_matrix(val_labels, val_pred, "BERT", return_fig=True)
                st.pyplot(fig_cm)
                plt.close(fig_cm)
            
            with col2:
                st.subheader("ROC Curve")
                fig_roc = plot_roc_curve(val_labels, val_proba, "BERT", return_fig=True)
                st.pyplot(fig_roc)
                plt.close(fig_roc)
            
            # Classification Report
            with st.expander("📋 Detailed Classification Report"):
                report_df = get_classification_report(val_labels, val_pred, 
                                                     target_names=['Human', 'AI-Generated'])
                st.dataframe(report_df, width='stretch')
            
            results['bert'] = metrics
            
            # Save model
            if save_models:
                os.makedirs('models', exist_ok=True)
                torch.save(model.model.state_dict(), 'models/bert_model.pt')
                st.info("💾 Model saved to `models/bert_model.pt`")
            
            # Save metrics and visualizations using evaluation.py
            if save_metrics:
                os.makedirs('results', exist_ok=True)
                pd.DataFrame([metrics]).to_csv('results/bert_metrics.csv', index=False)
                
                # Save classification report
                report_df.to_csv('results/bert_classification_report.csv')
                
                # Save visualizations using evaluation.py functions
                plot_confusion_matrix(val_labels, val_pred, "BERT", 
                                    save_path='results/bert_confusion_matrix.png')
                plot_roc_curve(val_labels, val_proba, "BERT", 
                             save_path='results/bert_roc_curve.png')
                
                st.info("📊 Metrics and visualizations saved to `results/`")
                
        except Exception as e:
            st.error(f"❌ BERT training failed: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Train RoBERTa Model
    if train_roberta:
        st.markdown("---")
        st.subheader("🟣 Training RoBERTa Model")
        progress_bar = st.progress(0, text="Initializing RoBERTa...")
        
        try:
            progress_bar.progress(10, text="Loading RoBERTa model...")
            model = RoBERTaClassifier()
            
            progress_bar.progress(20, text="Starting training...")
            st.info(f"Training for {epochs} epochs with batch size {batch_size}")
            
            # Train with progress updates
            model.train(
                train_texts, train_labels,
                val_texts, val_labels,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate
            )
            
            progress_bar.progress(90, text="Evaluating...")
            
            # Evaluate
            val_pred = model.predict(val_texts, batch_size=batch_size)
            val_proba = model.predict_proba(val_texts, batch_size=batch_size)
            
            from src.evaluation import (
                evaluate_model, 
                plot_confusion_matrix, 
                plot_roc_curve,
                get_classification_report
            )
            import matplotlib.pyplot as plt
            
            metrics = evaluate_model(val_labels, val_pred, val_proba, "RoBERTa")
            
            progress_bar.progress(100, text="Complete!")
            
            st.success(f"✅ RoBERTa trained! Validation Accuracy: {metrics['accuracy']:.4f}")
            
            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            col2.metric("Precision", f"{metrics['precision']:.4f}")
            col3.metric("Recall", f"{metrics['recall']:.4f}")
            col4.metric("F1 Score", f"{metrics['f1']:.4f}")
            if 'auc_roc' in metrics:
                col5.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")
            
            # Visualizations using evaluation.py functions
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Confusion Matrix")
                fig_cm = plot_confusion_matrix(val_labels, val_pred, "RoBERTa", return_fig=True)
                st.pyplot(fig_cm)
                plt.close(fig_cm)
            
            with col2:
                st.subheader("ROC Curve")
                fig_roc = plot_roc_curve(val_labels, val_proba, "RoBERTa", return_fig=True)
                st.pyplot(fig_roc)
                plt.close(fig_roc)
            
            # Classification Report
            with st.expander("📋 Detailed Classification Report"):
                report_df = get_classification_report(val_labels, val_pred, 
                                                     target_names=['Human', 'AI-Generated'])
                st.dataframe(report_df, width='stretch')
            
            results['roberta'] = metrics
            
            # Save model
            if save_models:
                os.makedirs('models', exist_ok=True)
                torch.save(model.model.state_dict(), 'models/roberta_model.pt')
                st.info("💾 Model saved to `models/roberta_model.pt`")
            
            # Save metrics and visualizations using evaluation.py
            if save_metrics:
                os.makedirs('results', exist_ok=True)
                pd.DataFrame([metrics]).to_csv('results/roberta_metrics.csv', index=False)
                
                # Save classification report
                report_df.to_csv('results/roberta_classification_report.csv')
                
                # Save visualizations using evaluation.py functions
                plot_confusion_matrix(val_labels, val_pred, "RoBERTa", 
                                    save_path='results/roberta_confusion_matrix.png')
                plot_roc_curve(val_labels, val_proba, "RoBERTa", 
                             save_path='results/roberta_roc_curve.png')
                
                st.info("📊 Metrics and visualizations saved to `results/`")
                
        except Exception as e:
            st.error(f"❌ RoBERTa training failed: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Summary
    if results:
        st.markdown("---")
        st.subheader("📊 Training Summary")
        
        summary_df = pd.DataFrame(results).T
        st.dataframe(summary_df, width='stretch')
        
        # Model comparison visualization if multiple models trained
        if len(results) > 1:
            st.subheader("📈 Model Comparison")
            
            from src.evaluation import compare_models
            import matplotlib.pyplot as plt
            
            # Use compare_models from evaluation.py
            _, fig = compare_models(results, return_fig=True)
            st.pyplot(fig)
            plt.close(fig)
            
            # Save comparison plot
            if save_metrics:
                compare_models(results, save_path='results/model_comparison.png')
                st.info("💾 Comparison plot saved to `results/model_comparison.png`")
        
        st.success("🎉 Training complete! You can now use the trained models in the Classification page.")
        st.info("💡 Tip: Reload the app or navigate to another page to refresh the model cache.")


if __name__ == "__main__":
    main()
