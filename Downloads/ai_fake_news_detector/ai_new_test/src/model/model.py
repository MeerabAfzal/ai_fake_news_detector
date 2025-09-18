import torch
import torch.nn as nn
import torch.nn.functional as F

class FakeNewsDetector(nn.Module):
    """
    LSTM-based model for fake news detection
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers=1, bidirectional=True, dropout=0.3, pad_idx=0):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=n_layers, 
                            bidirectional=bidirectional, 
                            dropout=dropout if n_layers > 1 else 0,
                            batch_first=True)
        
        # Determine the input size for the fully connected layer
        # If bidirectional, we need to multiply by 2
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Fully connected layer
        self.fc = nn.Linear(fc_input_dim, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        """
        Forward pass through the model
        
        Args:
            text: Tensor of shape [batch_size, seq_length]
            
        Returns:
            output: Tensor of shape [batch_size, output_dim]
        """
        # text = [batch_size, seq_length]
        
        # Embed the text
        embedded = self.dropout(self.embedding(text))
        # embedded = [batch_size, seq_length, embedding_dim]
        
        # Pass through LSTM
        output, (hidden, cell) = self.lstm(embedded)
        # output = [batch_size, seq_length, hidden_dim * n_directions]
        # hidden = [n_layers * n_directions, batch_size, hidden_dim]
        
        # If bidirectional, concatenate the final forward and backward hidden layers
        if self.lstm.bidirectional:
            # hidden = [n_layers * 2, batch_size, hidden_dim]
            # Take the final hidden state from last layer
            hidden_fwd = hidden[-2, :, :]  # Forward direction from last layer
            hidden_bwd = hidden[-1, :, :]  # Backward direction from last layer
            # Concatenate the two directions
            hidden = torch.cat((hidden_fwd, hidden_bwd), dim=1)
        else:
            # hidden = [n_layers, batch_size, hidden_dim]
            # Take the final hidden state from last layer
            hidden = hidden[-1, :, :]
            
        # hidden = [batch_size, hidden_dim * n_directions]
            
        # Apply dropout before the fully connected layer
        hidden = self.dropout(hidden)
        
        # Pass through fully connected layer
        output = self.fc(hidden)
        # output = [batch_size, output_dim]
        
        return output
        
    def predict(self, text, threshold=0.5):
        """
        Make a prediction using the model
        
        Args:
            text: Tensor of shape [batch_size, seq_length]
            threshold: Threshold for binary classification
            
        Returns:
            predictions: Binary predictions (0 or 1)
            probabilities: Probabilities of the positive class
        """
        with torch.no_grad():
            # Get model output
            output = self(text)
            
            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(output).squeeze(1)
            
            # Apply threshold to get binary predictions
            predictions = (probabilities >= threshold).int()
            
            return predictions, probabilities 