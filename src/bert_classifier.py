"""
BERT-based Text Classifier
Fine-tunes BERT model for AI vs human text detection.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
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
                 device=None):
        """
        Initialize BERT classifier.
        
        Args:
            model_name: HuggingFace model name
            num_labels: Number of classification labels
            max_length: Maximum sequence length
            device: Device to use (cuda/cpu)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)
    
    def train(self,
             train_texts,
             train_labels,
             val_texts=None,
             val_labels=None,
             batch_size=16,
             epochs=3,
             learning_rate=2e-5,
             warmup_steps=500):
        """
        Train the BERT classifier.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            batch_size: Batch size
            epochs: Number of epochs
            learning_rate: Learning rate
            warmup_steps: Warmup steps for scheduler
        """
        # Create datasets
        train_dataset = TextDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_texts is not None:
            val_dataset = TextDataset(val_texts, val_labels, self.tokenizer, self.max_length)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        print(f"\nTraining BERT model for {epochs} epochs...")
        print(f"Total steps: {total_steps}")
        
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
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Train Accuracy: {train_accuracy:.4f}")
            
            # Validation
            if val_texts is not None:
                val_loss, val_accuracy = self._evaluate(val_loader)
                print(f"Val Loss: {val_loss:.4f}")
                print(f"Val Accuracy: {val_accuracy:.4f}")
    
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
            for batch in tqdm(data_loader, desc="Evaluating"):
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
