#!/usr/bin/env python3
"""
Example script demonstrating the speech classification system.

This script shows how to:
1. Load sample data
2. Train a speech classifier
3. Make predictions on new text
4. Evaluate the classifier's performance
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from speech_classifier import SpeechClassifier, get_sample_data


def main():
    print("=" * 70)
    print("Speech Classification System - Demo")
    print("=" * 70)
    print()
    
    # Load sample data
    print("1. Loading sample speech data...")
    texts, labels = get_sample_data()
    print(f"   Loaded {len(texts)} speech samples with {len(set(labels))} categories")
    print(f"   Categories: {', '.join(sorted(set(labels)))}")
    print()
    
    # Train classifier
    print("2. Training speech classifier...")
    classifier = SpeechClassifier(classifier_type='naive_bayes')
    metrics = classifier.train(texts, labels)
    print(f"   Training completed!")
    print(f"   Validation Accuracy: {metrics['accuracy']:.2%}")
    print()
    
    # Test with new examples
    print("3. Testing with new speech samples...")
    print()
    
    test_samples = [
        "Artificial intelligence is transforming the way we live and work. Machine learning algorithms can now recognize patterns in data with remarkable accuracy.",
        "You should definitely try this new restaurant downtown. The food is amazing and the prices are very reasonable. Don't miss out on this culinary experience!",
        "I once tried to fix my computer by hitting it. Surprisingly, it worked! But then I realized I had just turned off the monitor.",
        "To make a perfect cup of coffee, start by heating water to 195-205°F. Then, measure 2 tablespoons of ground coffee per 6 ounces of water.",
    ]
    
    expected_types = ['informative', 'persuasive', 'entertaining', 'demonstrative']
    
    for i, (sample, expected) in enumerate(zip(test_samples, expected_types), 1):
        print(f"Sample {i}:")
        print(f"  Text: \"{sample[:80]}...\"")
        
        # Get prediction
        prediction = classifier.predict(sample)
        print(f"  Predicted: {prediction}")
        print(f"  Expected: {expected}")
        
        # Get probabilities
        probabilities = classifier.predict_proba(sample)
        print(f"  Confidence scores:")
        for speech_type, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {speech_type}: {prob:.2%}")
        
        print()
    
    # Show summary
    print("=" * 70)
    print("Demo completed successfully!")
    print()
    print("You can use this classifier by:")
    print("1. Creating a SpeechClassifier instance")
    print("2. Training it with your own data using .train(texts, labels)")
    print("3. Making predictions with .predict(text)")
    print("=" * 70)


if __name__ == '__main__':
    main()
