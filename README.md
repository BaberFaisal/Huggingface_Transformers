# Mastering Pre-trained Transformers

This repository contains a hands-on exploration of the Hugging Face `transformers` library. The project moves from simplified, one-line pipelines to custom inference logic for sentiment analysis, language modeling, and text generation.

##  Overview

The primary notebook, `Huggingface_transformers.ipynb`, is designed as a deep dive into:
- **High-level Pipelines**: Instant deployment of models for NLP tasks.
- **Sentiment Classification**: Fine-tuned DistilBERT applications.
- **Masked Language Modeling (MLM)**: Using BERT to recover hidden information from text.
- **Probabilistic Text Generation**: Implementing the mathematical logic (Logits -> Softmax -> Sampling) behind modern generative models.

##  Requirements

To run this notebook, you will need:
- Python 3.8+
- PyTorch
- Hugging Face Transformers
- NumPy

Install the dependencies via pip:
```bash
pip install torch transformers huggingface_hub numpy
```

##  Key Implementations

### 1. Sentiment Analysis
The model classifies text into POSITIVE/NEGATIVE labels.
* **Exercise**: Analyzing noble house mottos from *A Song of Ice and Fire*.
* **Model**: `distilbert-base-uncased-finetuned-sst-2-english`.

### 2. Masked Language Modeling
Predicts the most probable tokens for a `[MASK]` placeholder.
* **Exercise**: Fact retrieval using BERT (e.g., "The Soviet Union was founded in [MASK].").
* **Model**: `bert-base-uncased`.

### 3. Manual Generation Pipeline
A custom loop that demonstrates how auto-regressive models generate text token-by-token.
- **Logits Processing**: Extracting raw model outputs.
- **Softmax Sampling**: Converting scores to probabilities and introducing randomness for diverse output.
- **Decoding**: Converting token IDs back into human-readable strings.

##  Models Used
| Task | Model Checkpoint |
| :--- | :--- |
| Sentiment | `distilbert-base-uncased-finetuned-sst-2-english` |
| MLM | `bert-base-uncased` |
| Generation | `gpt2` |

---
