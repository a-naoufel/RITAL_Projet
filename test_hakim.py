"""
Tests for the Hakim module.
"""

import unittest
from hakim import Hakim


class TestHakim(unittest.TestCase):
    """Test cases for the Hakim class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = Hakim()
    
    def test_initialization(self):
        """Test Hakim initialization."""
        self.assertEqual(self.processor.name, "Hakim")
        
        custom_processor = Hakim("CustomName")
        self.assertEqual(custom_processor.name, "CustomName")
    
    def test_greet(self):
        """Test the greet method."""
        greeting = self.processor.greet()
        self.assertEqual(greeting, "Hello, I am Hakim!")
    
    def test_process_text(self):
        """Test text processing functionality."""
        text = "Hello World"
        result = self.processor.process_text(text)
        
        self.assertEqual(result["original"], "Hello World")
        self.assertEqual(result["length"], 11)
        self.assertEqual(result["words"], 2)
        self.assertEqual(result["uppercase"], "HELLO WORLD")
        self.assertEqual(result["lowercase"], "hello world")
    
    def test_process_empty_text(self):
        """Test processing empty text."""
        result = self.processor.process_text("")
        
        self.assertEqual(result["original"], "")
        self.assertEqual(result["length"], 0)
        self.assertEqual(result["words"], 0)
    
    def test_process_text_invalid_input(self):
        """Test that invalid input raises ValueError."""
        with self.assertRaises(ValueError):
            self.processor.process_text(123)
        
        with self.assertRaises(ValueError):
            self.processor.process_text(None)


if __name__ == "__main__":
    unittest.main()
