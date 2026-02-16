"""
Unit tests for the speech classifier module.
"""

import sys
import os
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from speech_classifier import SpeechClassifier, get_sample_data


class TestSpeechClassifier(unittest.TestCase):
    """Test cases for the SpeechClassifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.texts, self.labels = get_sample_data()
    
    def test_initialization_naive_bayes(self):
        """Test classifier initialization with Naive Bayes."""
        classifier = SpeechClassifier(classifier_type='naive_bayes')
        self.assertEqual(classifier.classifier_type, 'naive_bayes')
        self.assertFalse(classifier.is_trained)
        self.assertIsNone(classifier.classes_)
    
    def test_initialization_logistic(self):
        """Test classifier initialization with Logistic Regression."""
        classifier = SpeechClassifier(classifier_type='logistic')
        self.assertEqual(classifier.classifier_type, 'logistic')
        self.assertFalse(classifier.is_trained)
    
    def test_initialization_invalid_type(self):
        """Test that invalid classifier type raises ValueError."""
        with self.assertRaises(ValueError):
            SpeechClassifier(classifier_type='invalid')
    
    def test_train(self):
        """Test training the classifier."""
        classifier = SpeechClassifier()
        metrics = classifier.train(self.texts, self.labels)
        
        self.assertTrue(classifier.is_trained)
        self.assertIn('accuracy', metrics)
        self.assertIn('classes', metrics)
        self.assertIsInstance(metrics['accuracy'], float)
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
    
    def test_predict_before_training(self):
        """Test that prediction before training raises RuntimeError."""
        classifier = SpeechClassifier()
        with self.assertRaises(RuntimeError):
            classifier.predict("This is a test.")
    
    def test_predict(self):
        """Test prediction after training."""
        classifier = SpeechClassifier()
        classifier.train(self.texts, self.labels)
        
        # Test with an informative speech
        prediction = classifier.predict(
            "The Earth orbits the Sun at an average distance of 93 million miles."
        )
        self.assertIn(prediction, ['informative', 'persuasive', 'entertaining', 'demonstrative'])
    
    def test_predict_proba_before_training(self):
        """Test that predict_proba before training raises RuntimeError."""
        classifier = SpeechClassifier()
        with self.assertRaises(RuntimeError):
            classifier.predict_proba("This is a test.")
    
    def test_predict_proba(self):
        """Test probability prediction."""
        classifier = SpeechClassifier()
        classifier.train(self.texts, self.labels)
        
        probabilities = classifier.predict_proba("This is a scientific fact.")
        
        # Check that probabilities are returned for all classes
        self.assertEqual(len(probabilities), 4)  # 4 speech types
        
        # Check that probabilities sum to approximately 1
        total_prob = sum(probabilities.values())
        self.assertAlmostEqual(total_prob, 1.0, places=5)
        
        # Check that all probabilities are between 0 and 1
        for prob in probabilities.values():
            self.assertGreaterEqual(prob, 0)
            self.assertLessEqual(prob, 1)
    
    def test_evaluate_before_training(self):
        """Test that evaluation before training raises RuntimeError."""
        classifier = SpeechClassifier()
        with self.assertRaises(RuntimeError):
            classifier.evaluate(["test"], ["informative"])
    
    def test_evaluate(self):
        """Test evaluation after training."""
        classifier = SpeechClassifier()
        classifier.train(self.texts, self.labels)
        
        # Use the same data for evaluation (just for testing)
        results = classifier.evaluate(self.texts[:5], self.labels[:5])
        
        self.assertIn('accuracy', results)
        self.assertIn('classification_report', results)
        self.assertIsInstance(results['accuracy'], float)
    
    def test_different_classifiers(self):
        """Test both classifier types produce valid results."""
        for clf_type in ['naive_bayes', 'logistic']:
            classifier = SpeechClassifier(classifier_type=clf_type)
            metrics = classifier.train(self.texts, self.labels)
            
            self.assertTrue(classifier.is_trained)
            self.assertGreaterEqual(metrics['accuracy'], 0)
            
            prediction = classifier.predict("This is a test sentence.")
            self.assertIsInstance(prediction, str)


class TestSampleData(unittest.TestCase):
    """Test cases for the sample data function."""
    
    def test_get_sample_data(self):
        """Test that sample data is returned correctly."""
        texts, labels = get_sample_data()
        
        # Check that we have data
        self.assertGreater(len(texts), 0)
        self.assertEqual(len(texts), len(labels))
        
        # Check that we have all four categories
        unique_labels = set(labels)
        expected_labels = {'informative', 'persuasive', 'entertaining', 'demonstrative'}
        self.assertEqual(unique_labels, expected_labels)
        
        # Check that texts are non-empty strings
        for text in texts:
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 0)
        
        # Check that labels are strings
        for label in labels:
            self.assertIsInstance(label, str)


if __name__ == '__main__':
    unittest.main()
