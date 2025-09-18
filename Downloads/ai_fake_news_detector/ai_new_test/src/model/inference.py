import os
import sys
import pickle
import torch
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import FakeNewsDetector
from preprocessing.utils import clean_text, spacy_tokenize, texts_to_sequences, pad_sequences

class FakeNewsPredictor:
    """
    Class for making predictions using the trained model
    """
    def __init__(self, model_path, vocab_path, metadata_path=None, device=None):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the trained model checkpoint
            vocab_path: Path to the vocabulary file
            metadata_path: Path to the metadata file
            device: Device to use for inference (defaults to CPU)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load vocabulary
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        
        # Load metadata if provided
        if metadata_path:
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
                self.max_seq_length = self.metadata.get('max_seq_length', 100)
        else:
            self.metadata = None
            self.max_seq_length = 100
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model with the same parameters
        self.model = self._initialize_model()
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Model performance: Accuracy={checkpoint['test_metrics']['accuracy']:.4f}, F1={checkpoint['test_metrics']['f1']:.4f}")
    
    def _initialize_model(self):
        """
        Initialize the model with the same parameters as training
        """
        # Model parameters (using default values or from metadata)
        vocab_size = len(self.vocab)
        embedding_dim = 300
        hidden_dim = 256
        output_dim = 1
        n_layers = 2
        bidirectional = True
        dropout = 0.5
        
        return FakeNewsDetector(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout
        )
    
    def preprocess_text(self, text):
        """
        Preprocess a single text for prediction
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Tensor ready for model input
        """
        # Clean text
        cleaned_text = clean_text(text)
        
        # Tokenize
        tokens = spacy_tokenize(cleaned_text)
        
        # Convert to sequence
        sequence = texts_to_sequences([tokens], self.vocab)
        
        # Pad sequence
        padded_sequence = pad_sequences(sequence, max_length=self.max_seq_length)
        
        # Convert to tensor
        tensor = torch.tensor(padded_sequence, dtype=torch.long)
        
        return tensor.to(self.device)
    
    def predict(self, text, return_probability=False, threshold=0.5):
        """
        Make a prediction for a single text
        
        Args:
            text: Raw text to classify
            return_probability: Whether to return the probability
            threshold: Threshold for binary classification
            
        Returns:
            Prediction (FAKE or REAL) and optionally the probability
        """
        # Preprocess text
        tensor = self.preprocess_text(text)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(tensor).squeeze(1)
            probability = torch.sigmoid(output).item()
            prediction = "FAKE" if probability < threshold else "REAL"
        
        # Confidence score (distance from 0.5)
        confidence = abs(probability - 0.5) * 2  # Scale to 0-1
        
        if return_probability:
            return {
                "text": text,
                "prediction": prediction,
                "probability": probability,
                "confidence": confidence
            }
        else:
            return prediction
    
    def predict_batch(self, texts, threshold=0.5):
        """
        Make predictions for a batch of texts
        
        Args:
            texts: List of raw texts to classify
            threshold: Threshold for binary classification
            
        Returns:
            List of predictions
        """
        results = []
        
        for text in texts:
            result = self.predict(text, return_probability=True, threshold=threshold)
            results.append(result)
        
        return results

def load_predictor(model_dir='checkpoints'):
    """
    Helper function to load the predictor from saved files
    """
    model_path = os.path.join(model_dir, 'best_model.pt')
    vocab_path = os.path.join('data', 'processed', 'vocab.pkl')
    metadata_path = os.path.join('data', 'processed', 'metadata.pkl')
    
    return FakeNewsPredictor(model_path, vocab_path, metadata_path)

# Example usage
if __name__ == "__main__":
    # Sample text
    sample_texts = [
        "Scientists confirm COVID-19 originated in lab.",
        "NASA announces new Mars rover mission."
    ]
    
    try:
        # Try to load predictor
        predictor = load_predictor()
        
        # Make predictions
        for text in sample_texts:
            result = predictor.predict(text, return_probability=True)
            print(f"Text: {text}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print()
            
    except FileNotFoundError:
        print("Model files not found. Please train the model first.")
        print("Run: python src/training/train_model.py") 