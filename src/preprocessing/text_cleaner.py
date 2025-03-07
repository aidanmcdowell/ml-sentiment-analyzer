import re
import string
import unicodedata
from typing import List, Optional

class TextCleaner:
    """
    Class for cleaning and preprocessing text data for sentiment analysis.
    """
    
    def __init__(self, language: str = "english"):
        """
        Initialize the TextCleaner.
        
        Args:
            language: The language of the text to be processed.
        """
        self.language = language
        self.initialize_resources()
        
    def initialize_resources(self):
        """Initialize NLP resources (stopwords, etc.)."""
        # In a real implementation, we would load NLTK or spaCy resources here
        # For demonstration, we'll just define some common English stopwords
        self.stopwords = {
            "a", "an", "the", "and", "but", "or", "for", "nor", "on", "at", "to", "by",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "i", "you", "he", "she", "it", "we", "they", "this",
            "that", "these", "those", "am", "is", "are", "of", "in", "from", "with"
        }
        
    def clean(self, text: str, remove_stopwords: bool = False) -> str:
        """
        Clean and preprocess the input text.
        
        Args:
            text: The text to be cleaned.
            remove_stopwords: Whether to remove stopwords.
            
        Returns:
            The cleaned text.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords if requested
        if remove_stopwords:
            text = self.remove_stopwords(text)
            
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from the text.
        
        Args:
            text: The text to remove stopwords from.
            
        Returns:
            The text with stopwords removed.
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords]
        return ' '.join(filtered_words)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the text into words.
        
        Args:
            text: The text to tokenize.
            
        Returns:
            A list of tokens (words).
        """
        # Basic tokenization by whitespace
        # In a real implementation, we would use NLTK or spaCy tokenizers
        return text.split()
    
    def process_batch(self, texts: List[str], remove_stopwords: bool = False) -> List[str]:
        """
        Process a batch of texts.
        
        Args:
            texts: List of texts to clean.
            remove_stopwords: Whether to remove stopwords.
            
        Returns:
            List of cleaned texts.
        """
        return [self.clean(text, remove_stopwords) for text in texts] 