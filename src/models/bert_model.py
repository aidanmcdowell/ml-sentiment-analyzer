import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from typing import Dict, Any, List, Union

class BERTSentimentModel:
    """
    BERT-based sentiment analysis model using the HuggingFace Transformers library.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", device: str = None):
        """
        Initialize the BERT sentiment analysis model.
        
        Args:
            model_name: The name of the pre-trained BERT model to use.
            device: The device to run inference on (cuda or cpu).
        """
        self.model_name = model_name
        
        # Set the device (GPU if available, otherwise CPU)
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load pre-trained model and tokenizer
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=3,  # positive, negative, neutral
                output_attentions=False,
                output_hidden_states=False,
            )
            self.model.to(self.device)
            self.model.eval()  # Set the model to evaluation mode
            print(f"Successfully loaded BERT model {model_name} on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fall back to a simple model for demonstration
            self._initialize_dummy_model()
    
    def _initialize_dummy_model(self):
        """Initialize a dummy model for demonstration purposes."""
        print("Initializing dummy sentiment model for demonstration")
        self.dummy_model = True
    
    def preprocess(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess the input text for the BERT model.
        
        Args:
            text: The input text to analyze.
            
        Returns:
            The tokenized input ready for the model.
        """
        if hasattr(self, 'dummy_model'):
            return {"input_text": text}
            
        # Tokenize the text
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move inputs to the correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict the sentiment of the input text.
        
        Args:
            text: The input text to analyze.
            
        Returns:
            A dictionary containing the sentiment label, confidence score,
            and individual class scores.
        """
        if hasattr(self, 'dummy_model'):
            # Return dummy predictions for demonstration
            sentiment_scores = self._dummy_predict(text)
        else:
            # Preprocess the text
            inputs = self.preprocess(text)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get the logits (raw prediction values)
            logits = outputs.logits
            
            # Convert logits to probabilities using softmax
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            
            # Convert to numpy for easier processing
            sentiment_scores = probabilities.cpu().numpy()[0]
        
        # Map indices to sentiment labels
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        
        # Get the predicted class and confidence
        predicted_class = np.argmax(sentiment_scores)
        confidence = sentiment_scores[predicted_class]
        
        # Format the result
        result = {
            "label": label_map[predicted_class],
            "confidence": float(confidence),
            "scores": {
                "negative": float(sentiment_scores[0]),
                "neutral": float(sentiment_scores[1]),
                "positive": float(sentiment_scores[2])
            }
        }
        
        return result
    
    def _dummy_predict(self, text: str) -> np.ndarray:
        """
        Generate dummy predictions based on simple rules.
        Used for demonstration when the real model isn't available.
        
        Args:
            text: The input text.
            
        Returns:
            Numpy array of sentiment scores.
        """
        # Count positive and negative words
        positive_words = ["good", "great", "excellent", "amazing", "love", "happy", "best"]
        negative_words = ["bad", "terrible", "awful", "hate", "worst", "disappointed", "poor"]
        
        # Simple rule-based scoring
        text_lower = text.lower()
        pos_count = sum(word in text_lower for word in positive_words)
        neg_count = sum(word in text_lower for word in negative_words)
        
        # Base scores
        scores = np.array([0.2, 0.6, 0.2])  # [negative, neutral, positive]
        
        # Adjust based on word counts
        if pos_count > neg_count:
            scores = np.array([0.1, 0.3, 0.6])
        elif neg_count > pos_count:
            scores = np.array([0.6, 0.3, 0.1])
        
        return scores
    
    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for a batch of texts.
        
        Args:
            texts: List of input texts to analyze.
            
        Returns:
            List of dictionaries containing sentiment analysis results.
        """
        return [self.predict(text) for text in texts] 