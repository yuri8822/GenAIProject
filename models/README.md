# Models Directory

This directory stores trained model checkpoints.

## Structure

```
models/
├── baseline_model.pkl       # Logistic Regression model
├── bert_model/             # BERT model directory
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files
└── roberta_model/          # RoBERTa model directory
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer files
```

## Usage

Models are automatically saved here during training.

To load a saved model:

```python
# Baseline
from src.baseline_model import BaselineClassifier
model = BaselineClassifier()
model.load_model('models/baseline_model.pkl')

# BERT
from src.bert_classifier import BERTClassifier
model = BERTClassifier()
model.load_model('models/bert_model')

# RoBERTa
from src.roberta_classifier import RoBERTaClassifier
model = RoBERTaClassifier()
model.load_model('models/roberta_model')
```
