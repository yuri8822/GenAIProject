"""
AI Text Detection Project
Detecting AI-Generated vs. Human-Written Text Using Transformer-Based Models
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from . import preprocessing
from . import baseline_model
from . import bert_classifier
from . import roberta_classifier
from . import evaluation

__all__ = [
    'preprocessing',
    'baseline_model',
    'bert_classifier',
    'roberta_classifier',
    'evaluation'
]
