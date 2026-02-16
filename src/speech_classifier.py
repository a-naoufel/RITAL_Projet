"""
Speech Classification Module

This module provides functionality to classify speech into different categories
such as formal, informal, narrative, persuasive, etc.
"""

from typing import List, Dict, Tuple, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


class SpeechClassifier:
    """
    A classifier for categorizing speech text into different types.
    
    Supported speech types:
    - Informative: Presents facts and information
    - Persuasive: Attempts to convince or influence
    - Entertaining: Aims to amuse or engage
    - Demonstrative: Shows how to do something
    """
    
    def __init__(self, classifier_type='naive_bayes'):
        """
        Initialize the speech classifier.
        
        Args:
            classifier_type (str): Type of classifier to use ('naive_bayes' or 'logistic')
        """
        self.classifier_type = classifier_type
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        if classifier_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif classifier_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        self.is_trained = False
        self.classes_ = None
    
    def train(self, texts: List[str], labels: List[str]) -> Dict[str, float]:
        """
        Train the classifier on provided speech texts and labels.
        
        Args:
            texts (List[str]): List of speech texts
            labels (List[str]): List of corresponding labels
            
        Returns:
            Dict[str, float]: Training metrics including accuracy
        """
        # Convert texts to TF-IDF features
        X = self.vectorizer.fit_transform(texts)
        
        # Store unique classes
        self.classes_ = np.unique(labels)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, labels, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on validation set
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        return {
            'accuracy': accuracy,
            'classes': list(self.classes_)
        }
    
    def predict(self, text: str) -> str:
        """
        Predict the speech type for a given text.
        
        Args:
            text (str): Speech text to classify
            
        Returns:
            str: Predicted speech type
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        X = self.vectorizer.transform([text])
        prediction = self.model.predict(X)
        return prediction[0]
    
    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Predict probabilities for each speech type.
        
        Args:
            text (str): Speech text to classify
            
        Returns:
            Dict[str, float]: Dictionary mapping speech types to probabilities
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        X = self.vectorizer.transform([text])
        probabilities = self.model.predict_proba(X)[0]
        
        return {
            class_name: float(prob)
            for class_name, prob in zip(self.classes_, probabilities)
        }
    
    def evaluate(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """
        Evaluate the classifier on test data.
        
        Args:
            texts (List[str]): List of speech texts
            labels (List[str]): List of true labels
            
        Returns:
            Dict[str, any]: Evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")
        
        X = self.vectorizer.transform(texts)
        y_pred = self.model.predict(X)
        
        accuracy = accuracy_score(labels, y_pred)
        report = classification_report(labels, y_pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'classification_report': report
        }


def get_sample_data() -> Tuple[List[str], List[str]]:
    """
    Generate sample speech data for demonstration purposes.
    
    Returns:
        Tuple[List[str], List[str]]: Texts and their corresponding labels
    """
    texts = [
        # Informative speeches
        "The human brain contains approximately 86 billion neurons. Each neuron can form thousands of connections with other neurons.",
        "Climate change refers to long-term shifts in global temperatures and weather patterns. These shifts are primarily caused by human activities.",
        "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum in 1991.",
        "The process of photosynthesis converts light energy into chemical energy. Plants use this process to produce glucose from carbon dioxide and water.",
        "Democracy is a system of government where citizens exercise power by voting. It originated in ancient Athens around 500 BCE.",
        
        # Persuasive speeches
        "We must act now to protect our environment. Future generations are counting on us to make the right decisions today.",
        "Investing in education is the smartest investment a nation can make. It yields returns that benefit society for generations.",
        "Join us in this important cause. Together, we can make a real difference in our community and change lives for the better.",
        "This product will revolutionize your daily routine. Don't miss this opportunity to transform your life and save time.",
        "Vote for change, vote for progress. Our community deserves better leadership and a brighter future for all.",
        
        # Entertaining speeches
        "So I walked into this coffee shop, and you won't believe what happened next. The barista looked at me and said my usual order before I even spoke!",
        "Let me tell you about the time I tried to bake a cake. Long story short, the fire department now knows me by name.",
        "Have you ever noticed how cats act like they own the place? My cat treats me like I'm the pet and she's the owner.",
        "Dating in the modern age is wild. I once had someone cancel our date because their houseplant looked sad.",
        "My attempt at assembling IKEA furniture was like playing a Swedish version of a puzzle game, but harder.",
        
        # Demonstrative speeches
        "First, preheat your oven to 350 degrees. Then, mix the flour, sugar, and eggs in a large bowl until well combined.",
        "To change a tire, start by loosening the lug nuts. Next, use the jack to lift the vehicle off the ground.",
        "Begin by opening the application. Click on the settings icon in the top right corner, then select preferences from the menu.",
        "Step one: gather all your ingredients and tools. Step two: measure the ingredients precisely according to the recipe.",
        "To perform CPR, first check if the person is responsive. Then, call emergency services and begin chest compressions at 100-120 per minute.",
    ]
    
    labels = [
        'informative', 'informative', 'informative', 'informative', 'informative',
        'persuasive', 'persuasive', 'persuasive', 'persuasive', 'persuasive',
        'entertaining', 'entertaining', 'entertaining', 'entertaining', 'entertaining',
        'demonstrative', 'demonstrative', 'demonstrative', 'demonstrative', 'demonstrative',
    ]
    
    return texts, labels
