# AI-Text Detection Project

## Detecting AI-Generated vs. Human-Written Text Using Transformer-Based Models

### Overview
This project builds a text classifier to distinguish between human-written and AI-generated text using transformer-based models. The focus is on benchmarking multiple approaches and evaluating robustness against paraphrased AI text.

### Problem Statement
Existing AI-text detectors are unreliable, often failing when text is slightly rephrased. This project aims to create a lightweight yet effective detection system for academic and professional environments.

### Objectives
- Train and compare 2–3 classifiers (Logistic Regression with stylometry, BERT, RoBERTa)
- Evaluate performance using accuracy, precision, recall, and F1 scores
- Test robustness against paraphrased AI-generated text

### Datasets
- **HC3 Dataset**: Human vs. ChatGPT responses
- **Kaggle AI vs. Human Text**: Classification dataset
- Both datasets are publicly available and English-only

### Project Structure
```
.
├── data/                  # Raw and processed datasets
├── models/                # Saved model checkpoints
├── notebooks/             # Jupyter notebooks for experiments
├── src/                   # Source code
│   ├── preprocessing.py   # Data cleaning and tokenization
│   ├── baseline_model.py  # Logistic Regression classifier
│   ├── bert_classifier.py # BERT-based model
│   ├── roberta_classifier.py # RoBERTa-based model
│   └── evaluation.py      # Evaluation metrics and testing
├── results/               # Outputs, metrics, plots
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### Setup Instructions

1. **Clone/Navigate to project directory**
   ```bash
   cd "F:\CloudDrive\UNI\Semester 9\GENAI\Project"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download datasets**
   - Place HC3 and Kaggle datasets in the `data/` folder
   - See `data/README.md` for details

4. **Run preprocessing**
   ```bash
   python src/preprocessing.py
   ```

5. **Train models**
   ```bash
   # Baseline model
   python src/baseline_model.py
   
   # BERT classifier
   python src/bert_classifier.py
   
   # RoBERTa classifier
   python src/roberta_classifier.py
   ```

6. **Evaluate models**
   ```bash
   python src/evaluation.py
   ```

### Methodology

1. **Preprocessing**: Clean, tokenize, and normalize text data
2. **Model Training**:
   - Logistic Regression with stylometry features (baseline)
   - BERT-based classifier
   - RoBERTa classifier
3. **Evaluation**: Standard metrics + adversarial testing with paraphrased samples
4. **Analysis**: Compare strengths, weaknesses, and practical applicability

### Timeline (8 Weeks)
- **Week 1–2**: Dataset preparation, baseline setup
- **Week 3–5**: Model training and evaluation
- **Week 6–7**: Robustness testing and analysis
- **Week 8**: Report writing and submission

### Expected Outcomes
- Benchmark of detection accuracy across multiple models
- Insights into robustness against paraphrased AI text
- Lightweight prototype detector (CLI or notebook-based)

### Tools & Technologies
- Python 3.8+
- HuggingFace Transformers
- Scikit-learn
- PyTorch
- Pandas, NumPy, Matplotlib

### License
Academic use only

### Contributors
[Your Name]
