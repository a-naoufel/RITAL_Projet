# Sentiment Classification on Movie Reviews

A comparative NLP project for **binary sentiment classification** on English movie reviews.  
The goal is to predict whether a review is **positive** or **negative**, and generate a submission file for an unlabeled test set.

This repository explores a progression of models, from strong classical baselines to modern pretrained encoders.

## Project goals

- Build a complete sentiment classification pipeline
- Compare several model families on the same dataset
- Study the effect of:
  - sparse text representations
  - pretrained embeddings
  - sentence encoders
  - transformer fine-tuning
- select the best model for final prediction on the test set

## Models explored

The experiments are organized in the following order:

1. **TF–IDF + LinearSVC**  
   Strong classical baseline for text classification

2. **TF–IDF word + char ensemble**  
   Combination of word n-grams and character n-grams

3. **fastText supervised**  
   Lightweight and efficient text classifier with subword information

4. **SetFit**  
   Sentence-transformer-based few-shot / small-data classification approach

5. **DistilBERT SST-2**  
   Pretrained transformer already adapted to sentiment-related tasks

6. **DeBERTa-v3-base**  
   More powerful pretrained transformer for fine-tuning

7. **Optional: ModernBERT-base**  
   A more recent encoder to test as an additional modern baseline

## Why these models?

This project is not only about reaching the best score. It is also about understanding **which models are relevant for this task and why**.

The selected approaches cover different levels of complexity:

- **classical sparse models**: strong on small and medium datasets
- **hybrid ensemble methods**: combine complementary signals
- **lightweight neural models**: efficient and easy to train
- **pretrained sentence models**: useful when labeled data is limited
- **full transformer fine-tuning**: powerful but more expensive

This makes the project a structured comparison between:
- simple vs complex models
- non-pretrained vs pretrained approaches
- fast baselines vs heavier fine-tuned encoders

## Dataset

Training data is organized as text files:

```text
dataset/
  movies1000/
    pos/   # positive reviews
    neg/   # negative reviews
  testSentiment.txt
