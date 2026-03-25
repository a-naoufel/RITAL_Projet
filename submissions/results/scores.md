# Validation Accuracy Tracker

Split: 80/20 stratified, seed=42  
Metric: accuracy on validation set (400 examples)

| # | Model | Val Accuracy | Notes |
|---|-------|-------------|-------|
| 1 | NB-SVM (baseline) | — | to be completed |
| 2 | TF-IDF + LinearSVC (untuned) | 0.8725 | word n-grams (1,2) |
| 3 | TF-IDF + LinearSVC (tuned) | 0.8850 | C=4, word n-grams (1,2) |
| 4 | TF-IDF char n-grams | 0.8675 | char_wb, (3,5), C=0.5 |
| 5 | **TF-IDF word+char ensemble** | **0.8925** | alpha=0.9, best overall |
| 6 | fastText supervised | — | to be completed |
| 7 | SetFit | — | to be completed |
| 8 | DistilBERT base (fine-tuned) | 0.8225 | distilbert-base-uncased |
| 9 | DistilBERT SST-2 (fine-tuned) | 0.8575 | distilbert-base-uncased-finetuned-sst-2-english |
| 10 | RoBERTa-base (fine-tuned) | 0.8575 | appendix model |
| 11 | BiLSTM + GloVe | ~0.78 | appendix model |
| 12 | DeBERTa-v3-base | — | to be completed |

---
**Rejected approach:** line-level splitting → ~0.6397 (see models/tfidf_svc/train_tfidf_svc_lines.py)
