"""
Prepare HC3 dataset for AI text detection from local JSONL file.
HC3 (Human ChatGPT Comparison Corpus) contains human and ChatGPT responses.
"""

import os
import pandas as pd
import sys
import json
from sklearn.model_selection import train_test_split

def load_hc3_dataset(jsonl_path: str, output_dir: str = 'data'):
    """
    Load HC3 dataset from JSONL file and save to CSV format.
    
    Args:
        jsonl_path: Path to the HC3.jsonl file
        output_dir: Directory to save the processed dataset
    """
    print("=" * 60)
    print("HC3 DATASET PREPARATION")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n1. Reading HC3 dataset from: {jsonl_path}")
    
    if not os.path.exists(jsonl_path):
        print(f"   ✗ Error: File not found: {jsonl_path}")
        print("\n   Please ensure HC3.jsonl exists in the data folder.")
        return None
    
    print("   This may take a few minutes...")
    
    try:
        # Process the dataset
        all_texts = []
        all_labels = []
        line_count = 0
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                if line_count % 5000 == 0:
                    print(f"   Processing line {line_count}...")
                
                try:
                    item = json.loads(line.strip())
                    
                    # Human answers (label = 0)
                    human_answers = item.get('human_answers', [])
                    if isinstance(human_answers, list):
                        for answer in human_answers:
                            if answer and len(answer.strip()) > 50:  # Filter very short texts
                                all_texts.append(answer.strip())
                                all_labels.append(0)
                    
                    # ChatGPT answers (label = 1)
                    chatgpt_answers = item.get('chatgpt_answers', [])
                    if isinstance(chatgpt_answers, list):
                        for answer in chatgpt_answers:
                            if answer and len(answer.strip()) > 50:  # Filter very short texts
                                all_texts.append(answer.strip())
                                all_labels.append(1)
                                
                except json.JSONDecodeError as e:
                    print(f"   Warning: Skipping malformed JSON at line {line_count}")
                    continue
        
        print(f"   ✓ Dataset loaded successfully!")
        print(f"   Total lines processed: {line_count}")
        
    except Exception as e:
        print(f"   ✗ Error reading dataset: {e}")
        return None
    
    print(f"\n2. Dataset statistics:")
    print(f"   ✓ Collected {len(all_texts)} text samples")
    print(f"   - Human texts: {sum(1 for l in all_labels if l == 0)}")
    print(f"   - AI texts: {sum(1 for l in all_labels if l == 1)}")
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': all_texts,
        'label': all_labels
    })
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save full dataset
    full_path = os.path.join(output_dir, 'dataset.csv')
    df.to_csv(full_path, index=False)
    print(f"\n3. Saved full dataset to: {full_path}")
    
    # Create train/val/test splits
    print("\n4. Creating train/validation/test splits...")
    
    # First split: 80% train+val, 20% test
    train_val, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    # Second split: 75% train, 25% val (of the train_val set)
    train, val = train_test_split(train_val, test_size=0.25, random_state=42, stratify=train_val['label'])
    
    # Save splits
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)
    
    print(f"   - Train set: {len(train)} samples → {train_path}")
    print(f"   - Val set: {len(val)} samples → {val_path}")
    print(f"   - Test set: {len(test)} samples → {test_path}")
    
    # Print statistics
    print("\n5. Dataset Statistics:")
    print("\n   Train Set:")
    print(f"   - Human: {sum(train['label'] == 0)} ({sum(train['label'] == 0)/len(train)*100:.1f}%)")
    print(f"   - AI: {sum(train['label'] == 1)} ({sum(train['label'] == 1)/len(train)*100:.1f}%)")
    
    print("\n   Validation Set:")
    print(f"   - Human: {sum(val['label'] == 0)} ({sum(val['label'] == 0)/len(val)*100:.1f}%)")
    print(f"   - AI: {sum(val['label'] == 1)} ({sum(val['label'] == 1)/len(val)*100:.1f}%)")
    
    print("\n   Test Set:")
    print(f"   - Human: {sum(test['label'] == 0)} ({sum(test['label'] == 0)/len(test)*100:.1f}%)")
    print(f"   - AI: {sum(test['label'] == 1)} ({sum(test['label'] == 1)/len(test)*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run preprocessing: python src/preprocessing.py")
    print("2. Train baseline model: python src/baseline_model.py")
    print("3. Train BERT/RoBERTa models")
    
    return df


if __name__ == "__main__":
    # Get project paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    jsonl_path = os.path.join(data_dir, 'HC3.jsonl')
    
    print("HC3 Dataset Preparation Script")
    print(f"Reading from: {jsonl_path}")
    print(f"Output will be saved to: {data_dir}\n")
    
    # Load and prepare dataset
    dataset = load_hc3_dataset(jsonl_path, data_dir)
    
    if dataset is not None:
        print("\n✓ Dataset ready for training!")
    else:
        print("\n✗ Dataset preparation failed. Please check errors above.")
        sys.exit(1)
