# Dataset Information

## Required Datasets

### 1. HC3 Dataset
**Source**: [Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)

**Description**: Human-ChatGPT Comparison Corpus containing:
- Human-written answers
- ChatGPT-generated responses
- Same questions answered by both

**Download Instructions**:
```python
from datasets import load_dataset
dataset = load_dataset("Hello-SimpleAI/HC3", "all")
# Save to CSV
dataset['train'].to_csv('hc3_train.csv', index=False)
```

Or download directly from HuggingFace and place in this directory.

**Expected Format**:
- `text` column: The text content
- `label` column: 0 for human, 1 for AI-generated

---

### 2. Kaggle AI vs Human Text Dataset
**Source**: [Kaggle - AI vs Human Text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text)

**Description**: Collection of texts from various sources labeled as human or AI-generated.

**Download Instructions**:
1. Visit the Kaggle dataset page
2. Download the CSV file
3. Place it in this `data/` directory
4. Rename to `kaggle_ai_human.csv` (or update preprocessing script)

**Expected Format**:
- `text` column: The text content
- `generated` or `label` column: 0/human or 1/ai

---

## Directory Structure

```
data/
├── README.md                 # This file
├── hc3_train.csv            # HC3 dataset (download)
├── kaggle_ai_human.csv      # Kaggle dataset (download)
├── processed/               # Preprocessed data (auto-generated)
│   ├── train.pkl
│   ├── val.pkl
│   └── test.pkl
└── paraphrased/             # For robustness testing (optional)
    └── paraphrased_samples.csv
```

---

## Data Preprocessing

After downloading datasets, run the preprocessing script:

```bash
python src/preprocessing.py
```

This will:
1. Load and clean text data
2. Extract stylometric features
3. Split into train/val/test sets
4. Save processed data to `data/processed/`

---

## Sample Data Format

Your CSV files should have at minimum:

| text | label |
|------|-------|
| "This is a human-written text..." | 0 |
| "This is an AI-generated text..." | 1 |

---

## Paraphrasing for Robustness Testing

To test model robustness, you'll need paraphrased versions of AI-generated texts.

**Methods**:
1. **ChatGPT/GPT-4**: "Paraphrase the following text: [text]"
2. **Pegasus Model**: Use HuggingFace's paraphrasing models
3. **Back-translation**: Translate to another language and back

**Example Script** (using GPT API):
```python
import openai

def paraphrase_text(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Paraphrase the following text while keeping the meaning."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content
```

---

## Notes

- **English Only**: Both datasets should contain English text only
- **Label Convention**: 0 = Human, 1 = AI-generated
- **Size Recommendation**: At least 10,000 samples for meaningful results
- **Balance**: Try to maintain roughly equal distribution of human/AI texts

---

## Dataset Statistics

After loading, verify your dataset:

```python
import pandas as pd

df = pd.read_csv('hc3_train.csv')
print(f"Total samples: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}")
print(f"Average text length: {df['text'].str.len().mean():.0f} chars")
```

---

## Citation

If using HC3 dataset:
```
@article{guo2023close,
  title={How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection},
  author={Guo, Biyang and Zhang, Xin and Wang, Ziyuan and Jiang, Minqi and Nie, Jinran and Ding, Yuxuan and Yue, Jianwei and Wu, Yupeng},
  journal={arXiv preprint arXiv:2301.07597},
  year={2023}
}
```
