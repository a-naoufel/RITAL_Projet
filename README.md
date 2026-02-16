# RITAL_Projet - Speech Classification System

A Python-based speech classification system that categorizes text into different speech types using machine learning.

## Overview

This project implements a text classification system that can identify different types of speech or communication styles. It uses Natural Language Processing (NLP) and machine learning techniques to automatically classify text into predefined categories.

## Speech Types

The classifier can distinguish between four main types of speech:

1. **Informative**: Speeches that present facts, data, and information objectively
   - Example: "The human brain contains approximately 86 billion neurons..."

2. **Persuasive**: Speeches that attempt to convince or influence the audience
   - Example: "We must act now to protect our environment..."

3. **Entertaining**: Speeches that aim to amuse, engage, or entertain the audience
   - Example: "Let me tell you about the time I tried to bake a cake..."

4. **Demonstrative**: Speeches that show how to do something or explain a process
   - Example: "First, preheat your oven to 350 degrees. Then, mix..."

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/a-naoufel/RITAL_Projet.git
cd RITAL_Projet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Example

To see the speech classifier in action, run the example script:

```bash
python example.py
```

This will:
- Load sample speech data
- Train the classifier
- Test it on new examples
- Display predictions with confidence scores

### Using in Your Code

Here's a simple example of how to use the speech classifier:

```python
from src.speech_classifier import SpeechClassifier, get_sample_data

# Load training data
texts, labels = get_sample_data()

# Create and train the classifier
classifier = SpeechClassifier(classifier_type='naive_bayes')
metrics = classifier.train(texts, labels)

# Make a prediction
text = "Climate change is a pressing issue that requires immediate action."
prediction = classifier.predict(text)
print(f"Speech type: {prediction}")

# Get probability scores
probabilities = classifier.predict_proba(text)
for speech_type, prob in probabilities.items():
    print(f"{speech_type}: {prob:.2%}")
```

### Training with Custom Data

You can train the classifier with your own data:

```python
# Prepare your data
my_texts = [
    "Text of first speech...",
    "Text of second speech...",
    # ... more texts
]

my_labels = [
    "informative",
    "persuasive",
    # ... corresponding labels
]

# Train the classifier
classifier = SpeechClassifier()
metrics = classifier.train(my_texts, my_labels)
print(f"Accuracy: {metrics['accuracy']:.2%}")
```

## API Reference

### SpeechClassifier

The main class for speech classification.

#### Constructor

```python
SpeechClassifier(classifier_type='naive_bayes')
```

Parameters:
- `classifier_type` (str): Type of classifier to use. Options: 'naive_bayes', 'logistic'

#### Methods

**train(texts, labels)**
Train the classifier on provided data.
- `texts` (List[str]): List of speech texts
- `labels` (List[str]): List of corresponding labels
- Returns: Dict with training metrics

**predict(text)**
Predict the speech type for given text.
- `text` (str): Speech text to classify
- Returns: str (predicted speech type)

**predict_proba(text)**
Get probability scores for each speech type.
- `text` (str): Speech text to classify
- Returns: Dict mapping speech types to probabilities

**evaluate(texts, labels)**
Evaluate classifier performance on test data.
- `texts` (List[str]): List of speech texts
- `labels` (List[str]): List of true labels
- Returns: Dict with evaluation metrics

## Project Structure

```
RITAL_Projet/
├── src/
│   ├── __init__.py
│   └── speech_classifier.py    # Main classifier implementation
├── tests/
│   ├── __init__.py
│   └── test_classifier.py      # Unit tests
├── example.py                   # Example usage script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── LICENSE                      # Apache 2.0 License
└── .gitignore                  # Git ignore file
```

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

## Technical Details

### Features

The classifier uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert text into numerical features. This approach:
- Captures word importance in documents
- Handles vocabulary of different sizes
- Provides good baseline performance for text classification

### Algorithms

Two classifier options are available:

1. **Naive Bayes** (default): Fast and effective for text classification
2. **Logistic Regression**: Often provides better accuracy with more data

### Performance

The classifier achieves good accuracy on the provided sample data. Performance depends on:
- Quality and quantity of training data
- Diversity of speech types
- Similarity between training and test data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Author

Created for the RITAL (Recherche d'Information et Traitement Automatique des Langues) project.