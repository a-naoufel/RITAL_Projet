# RITAL_Projet

A text processing project featuring the Hakim module for basic NLP operations.

## Features

### Hakim Module

The Hakim module provides basic text processing functionality including:
- Text analysis (word count, character count)
- Case conversion (uppercase/lowercase)
- Simple text processing operations

## Usage

```python
from hakim import Hakim

# Create a Hakim processor instance
processor = Hakim()

# Get a greeting
print(processor.greet())

# Process some text
text = "This is a sample text for processing."
result = processor.process_text(text)
print(result)
```

## Running the Module

```bash
python hakim.py
```

## Running Tests

```bash
python -m unittest test_hakim.py
```

## Requirements

- Python 3.6+

No external dependencies required for basic functionality.