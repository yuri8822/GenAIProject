"""
BERT-based Text Classifier
Fine-tunes BERT model for AI vs human text detection.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def seed_worker(worker_id):
    """Seed worker for reproducible DataLoader shuffling."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class TextDataset(Dataset):
    """Dataset for text classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BERTClassifier:
    """BERT-based classifier for AI text detection."""
    
    def __init__(self, 
                 model_name='bert-base-uncased',
                 num_labels=2,
                 max_length=512,
                 device=None,
                 dropout_rate=0.1,
                 seed=42,
                 use_fp16=False):
        """
        Initialize BERT classifier.
        
        Args:
            model_name: HuggingFace model name
            num_labels: Number of classification labels
            max_length: Maximum sequence length
            device: Device to use (cuda/cpu)
            dropout_rate: Dropout rate for regularization (default: 0.1)
            seed: Random seed for reproducibility (default: 42)
            use_fp16: Use mixed precision training (FP16) for faster training and lower memory (default: False)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.seed = seed
        self.use_fp16 = use_fp16 and torch.cuda.is_available()  # Only use FP16 if CUDA is available
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.training_history = []
        
        # Set seeds for reproducibility at initialization
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Enable deterministic mode for full reproducibility (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        print(f"Using device: {self.device}")
        print(f"Random seed: {seed}")
        print(f"Mixed precision (FP16): {'Enabled' if self.use_fp16 else 'Disabled'}")
        
        # Initialize tokenizer and model with specified dropout
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
            classifier_dropout=dropout_rate
        ).to(self.device)
        
        print(f"Model initialized with dropout rate: {dropout_rate}")
    
    def train(self,
             train_texts,
             train_labels,
             val_texts=None,
             val_labels=None,
             batch_size=16,
             epochs=3,
             learning_rate=2e-5,
             warmup_steps=500,
             weight_decay=0.01,
             max_grad_norm=1.0,
             early_stopping_patience=3):
        """
        Train the BERT classifier with regularization.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            batch_size: Batch size
            epochs: Number of epochs
            learning_rate: Learning rate
            warmup_steps: Warmup steps for scheduler
            weight_decay: L2 regularization coefficient (default: 0.01)
            max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
            early_stopping_patience: Stop if no improvement for N epochs (default: 3, 0 to disable)
        """
        # Input validation
        if len(train_texts) == 0:
            raise ValueError("Training data cannot be empty")
        if len(train_texts) != len(train_labels):
            raise ValueError(f"Mismatch between texts ({len(train_texts)}) and labels ({len(train_labels)})")
        if val_texts is not None and len(val_texts) != len(val_labels):
            raise ValueError(f"Mismatch between validation texts ({len(val_texts)}) and labels ({len(val_labels)})")
        
        # Validate early stopping requirements
        use_early_stopping = early_stopping_patience > 0 and val_texts is not None
        if early_stopping_patience > 0 and val_texts is None:
            print("Warning: Early stopping enabled but no validation data provided. Disabling early stopping.")
            use_early_stopping = False
        
        # Create generator for reproducible shuffling
        g = torch.Generator()
        g.manual_seed(self.seed)
        
        # Create datasets with reproducible DataLoaders
        train_dataset = TextDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g
        )
        
        if val_texts is not None:
            val_dataset = TextDataset(val_texts, val_labels, self.tokenizer, self.max_length)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size,
                worker_init_fn=seed_worker,
                generator=g
            )
        
        # Optimizer with weight decay (L2 regularization)
        optimizer = AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=1e-8
        )
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Initialize gradient scaler for mixed precision training
        scaler = torch.amp.GradScaler('cuda', enabled=self.use_fp16)
        
        # Early stopping variables
        best_val_loss = float('inf')
        best_val_accuracy = 0.0
        best_epoch = 0
        patience_counter = 0
        best_model_state = None
        
        # Reset training history
        self.training_history = []
        
        # Training loop
        print(f"\nTraining BERT model for {epochs} epochs...")
        print(f"Total steps: {total_steps}")
        print(f"Regularization: weight_decay={weight_decay}")
        print(f"Learning rate: {learning_rate}")
        print(f"Mixed precision (FP16): {'Enabled' if self.use_fp16 else 'Disabled'}")
        print(f"Early stopping patience: {early_stopping_patience} epochs" if use_early_stopping else "Early stopping: disabled")
        print(f"Random seed: {self.seed}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 40)
            
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in tqdm(train_loader, desc="Training"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                # Use automatic mixed precision if enabled
                with torch.amp.autocast('cuda', enabled=self.use_fp16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    logits = outputs.logits
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                train_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Train Accuracy: {train_accuracy:.4f}")
            
            # Track metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_accuracy': train_accuracy
            }
            
            # Validation
            if val_texts is not None:
                val_loss, val_accuracy = self._evaluate(val_loader)
                print(f"Val Loss: {val_loss:.4f}")
                print(f"Val Accuracy: {val_accuracy:.4f}")
                
                epoch_metrics['val_loss'] = val_loss
                epoch_metrics['val_accuracy'] = val_accuracy
                
                # Early stopping check
                if use_early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_val_accuracy = val_accuracy
                        best_epoch = epoch + 1
                        patience_counter = 0
                        # Clear old state and save new best (keep on same device to avoid memory transfers)
                        if best_model_state is not None:
                            del best_model_state
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                        print(f"✓ New best validation loss: {best_val_loss:.4f} (accuracy: {best_val_accuracy:.4f})")
                    else:
                        patience_counter += 1
                        print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")
                        
                        if patience_counter >= early_stopping_patience:
                            print(f"\nEarly stopping triggered after epoch {epoch + 1}")
                            print(f"Restoring best model from epoch {best_epoch}")
                            # Add current epoch metrics before breaking
                            self.training_history.append(epoch_metrics)
                            # Restore best model state
                            self.model.load_state_dict(best_model_state)
                            break
            
            self.training_history.append(epoch_metrics)
        
        # Clean up best_model_state to free memory
        if best_model_state is not None:
            del best_model_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"\nTraining completed!")
        if use_early_stopping:
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
        
        return self.training_history
    
    def _evaluate(self, data_loader):
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def predict(self, texts, batch_size=16):
        """
        Make predictions on new texts.
        
        Args:
            texts: List of texts
            batch_size: Batch size
            
        Returns:
            Array of predictions
        """
        dataset = TextDataset(texts, [0] * len(texts), self.tokenizer, self.max_length)
        data_loader = DataLoader(dataset, batch_size=batch_size)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(batch_predictions)
        
        return np.array(predictions)
    
    def predict_proba(self, texts, batch_size=16):
        """
        Get prediction probabilities.
        
        Args:
            texts: List of texts
            batch_size: Batch size
            
        Returns:
            Array of probabilities
        """
        dataset = TextDataset(texts, [0] * len(texts), self.tokenizer, self.max_length)
        data_loader = DataLoader(dataset, batch_size=batch_size)
        
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting probabilities"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probabilities.extend(probs)
        
        return np.array(probabilities)
    
    def predict_single(self, text):
        """
        Predict a single text sample.
        
        Args:
            text: Single text string
            
        Returns:
            Tuple of (prediction, probabilities)
        """
        self.model.eval()
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            prediction = np.argmax(probs)
        
        return prediction, probs
    
    def save_model(self, output_dir):
        """
        Save model and tokenizer.
        
        Args:
            output_dir: Directory to save model
        """
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
    
    def load_model(self, model_dir):
        """
        Load model and tokenizer.
        
        Args:
            model_dir: Directory containing saved model
        """
        self.model = BertForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        print(f"Model loaded from {model_dir}")


if __name__ == "__main__":
    print("BERT Classifier Script")
    print("=" * 60)
    print("\nThis script provides BERT-based classification for AI text detection.")
    print("\nExample usage:")
    print("  classifier = BERTClassifier()")
    print("  classifier.train(train_texts, train_labels, val_texts, val_labels)")
    print("  predictions = classifier.predict(test_texts)")
    print("\nRefer to the main notebook for complete training pipeline.")
