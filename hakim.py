"""
Hakim - A simple text processing module for RITAL project.

This module provides basic text processing functionality.
"""


class Hakim:
    """
    A text processor class for basic NLP operations.
    """
    
    def __init__(self, name="Hakim"):
        """
        Initialize the Hakim text processor.
        
        Args:
            name (str): The name of the processor instance.
        """
        self.name = name
    
    def greet(self):
        """
        Return a greeting message.
        
        Returns:
            str: A greeting message.
        """
        return f"Hello, I am {self.name}!"
    
    def process_text(self, text):
        """
        Process the input text by performing basic operations.
        
        Args:
            text (str): The input text to process.
        
        Returns:
            dict: A dictionary containing processed information.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        return {
            "original": text,
            "length": len(text),
            "words": len(text.split()),
            "uppercase": text.upper(),
            "lowercase": text.lower()
        }


def main():
    """Main function to demonstrate Hakim functionality."""
    processor = Hakim()
    print(processor.greet())
    
    sample_text = "This is a sample text for processing."
    result = processor.process_text(sample_text)
    print("\nProcessed text:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
