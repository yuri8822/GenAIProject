# Quick Start Guide

## AI Text Detection Project - Getting Started in 5 Steps

---

## Step 1: Install Dependencies

```powershell
# Navigate to project directory
cd "F:\CloudDrive\UNI\Semester 9\GENAI\Project"

# Install required packages
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

**Note**: If using GPU for BERT/RoBERTa, ensure PyTorch with CUDA is installed:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Step 2: Download Datasets

### Option A: HC3 Dataset (Recommended)
```python
from datasets import load_dataset
dataset = load_dataset("Hello-SimpleAI/HC3", "all")
dataset['train'].to_csv('data/hc3_train.csv', index=False)
```

### Option B: Kaggle Dataset
1. Visit: https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text
2. Download CSV and place in `data/` folder

See `data/README.md` for detailed instructions.

---

## Step 3: Prepare Your Data

Create a simple script to load and format your data:

```python
# create: prepare_data.py
import pandas as pd
from src.preprocessing import TextPreprocessor, split_dataset

# Load your dataset
df = pd.read_csv('data/hc3_train.csv')  # or your dataset
texts = df['text'].tolist()
labels = df['label'].tolist()  # 0=human, 1=AI

# Preprocess
preprocessor = TextPreprocessor()
cleaned_texts, labels, features_df = preprocessor.preprocess_dataset(
    texts, labels, extract_features=True
)

# Split
data_splits = split_dataset(cleaned_texts, labels, features_df)

# Save
import pickle
with open('data/processed/data_splits.pkl', 'wb') as f:
    pickle.dump(data_splits, f)

print("Data prepared and saved!")
```

Run it:
```powershell
python prepare_data.py
```

---

## Step 4: Train Models

### Quick Test (Baseline Only)
```python
from src.baseline_model import BaselineClassifier
import pickle

# Load data
with open('data/processed/data_splits.pkl', 'rb') as f:
    data = pickle.load(f)

# Train baseline
model = BaselineClassifier()
model.train(data['train']['features'], data['train']['labels'])
model.save_model('models/baseline_model.pkl')

# Evaluate
preds = model.predict(data['test']['features'])
from sklearn.metrics import accuracy_score
print(f"Accuracy: {accuracy_score(data['test']['labels'], preds):.4f}")
```

### Full Pipeline (All Models)
Open and run the Jupyter notebook:
```powershell
jupyter notebook notebooks/main_experiment.py
```

Or convert to notebook format and run cell by cell for better control.

---

## Step 5: Evaluate Results

### View Results
```python
import pandas as pd

# Load saved results
results = pd.read_csv('results/final_results.csv')
print(results)

# View plots
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('results/model_comparison.png')
plt.imshow(img)
plt.axis('off')
plt.show()
```

### Run Evaluation Script
```python
from src.evaluation import print_evaluation_report

# Evaluate any model
print_evaluation_report(
    y_true=test_labels,
    y_pred=predictions,
    y_proba=probabilities,
    model_name="My Model"
)
```

---

## Project Timeline (8 Weeks)

### Week 1-2: Setup & Baseline ✓
- [x] Install dependencies
- [x] Download datasets
- [x] Train baseline model
- [ ] Document baseline results

### Week 3-5: Deep Learning Models
- [ ] Train BERT classifier
- [ ] Train RoBERTa classifier
- [ ] Compare all models
- [ ] Tune hyperparameters

### Week 6-7: Robustness & Analysis
- [ ] Generate paraphrased samples
- [ ] Test robustness
- [ ] Error analysis
- [ ] Ablation studies

### Week 8: Final Report
- [ ] Compile results
- [ ] Write discussion
- [ ] Create visualizations
- [ ] Submit report

---

## Troubleshooting

### Issue: Out of Memory (GPU)
**Solution**: Reduce batch size
```python
bert_model.train(..., batch_size=8)  # instead of 16
```

### Issue: Slow Training
**Solution**: Use smaller model or reduce data
```python
# Use distilled models
BERTClassifier(model_name='distilbert-base-uncased')
```

### Issue: Import Errors
**Solution**: Reinstall packages
```powershell
pip install --upgrade transformers torch scikit-learn
```

### Issue: NLTK Data Not Found
**Solution**: Download required data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## Tips for Success

1. **Start Small**: Test with 1000 samples first before full dataset
2. **Save Checkpoints**: Save models after each epoch
3. **Monitor Training**: Use validation set to prevent overfitting
4. **GPU Usage**: Use Google Colab if no local GPU available
5. **Document Everything**: Keep notes on experiments and results

---

## Quick Commands Reference

```powershell
# Install dependencies
pip install -r requirements.txt

# Run preprocessing
python src/preprocessing.py

# Train baseline
python src/baseline_model.py

# Start Jupyter
jupyter notebook

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# View results
python -c "import pandas as pd; print(pd.read_csv('results/final_results.csv'))"
```

---

## Next Steps

1. ✓ Read through all code files to understand the structure
2. ✓ Download at least one dataset
3. ✓ Run baseline model to verify setup
4. ⏳ Train BERT/RoBERTa models
5. ⏳ Conduct robustness testing
6. ⏳ Write final report

---

## Resources

- **HuggingFace Transformers**: https://huggingface.co/docs/transformers
- **Scikit-learn**: https://scikit-learn.org/
- **PyTorch**: https://pytorch.org/tutorials/

---

## Getting Help

If you encounter issues:
1. Check error messages carefully
2. Review function docstrings
3. Test with small sample first
4. Verify data format matches expectations

Good luck with your project! 🚀
