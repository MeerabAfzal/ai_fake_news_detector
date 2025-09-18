from flask import Flask, request, jsonify, render_template
import os
import torch
import torch.nn as nn
import pickle
import re
import numpy as np

app = Flask(__name__)

# Model Architecture - needed to load the saved model
class AttentionBiLSTM(nn.Module):
    """Bidirectional LSTM with Attention mechanism for fake news detection."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, 
                 dropout=0.5, pad_idx=0, bidirectional=True):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout if n_layers > 1 else 0,
                            batch_first=True)
        
        self.attention = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        outputs, (hidden, cell) = self.lstm(embedded)
        attention_weights = torch.softmax(self.attention(outputs), dim=1)
        context = torch.sum(attention_weights * outputs, dim=1)
        output = self.fc(self.dropout(context))
        return output, attention_weights

class FakeNewsEnsembleModel(nn.Module):
    """Ensemble model containing all fold models as submodules"""
    def __init__(self, models):
        super().__init__()
        self.fold_models = nn.ModuleList(models)
    
    def forward(self, input_ids):
        """Forward pass for inference"""
        all_outputs = []
        all_attention = []
        
        for model in self.fold_models:
            output, attention = model(input_ids)
            all_outputs.append(torch.sigmoid(output))
            all_attention.append(attention)
        
        # Average the outputs and attention
        avg_output = torch.mean(torch.stack(all_outputs), dim=0)
        avg_attention = torch.mean(torch.stack(all_attention), dim=0)
        
        return avg_output, avg_attention

# Custom tokenizer class to replace BERT tokenizer
class CustomTokenizer:
    def __init__(self, vocab_file):
        # Load vocabulary from pickle file
        try:
            # Try the standard approach first
            with open(vocab_file, 'rb') as f:
                self.word_to_idx = pickle.load(f)
        except Exception as e1:
            try:
                # Try with encoding='bytes' which can help with Python 2/3 compatibility issues
                with open(vocab_file, 'rb') as f:
                    self.word_to_idx = pickle.load(f, encoding='bytes')
                
                # Convert byte keys to strings if needed
                if isinstance(next(iter(self.word_to_idx.keys())), bytes):
                    self.word_to_idx = {k.decode('utf-8'): v for k, v in self.word_to_idx.items()}
            except Exception as e2:
                try:
                    # Try with Latin-1 encoding as a last resort
                    with open(vocab_file, 'rb') as f:
                        self.word_to_idx = pickle.load(f, encoding='latin1')
                    
                    # Convert byte keys to strings if needed
                    if isinstance(next(iter(self.word_to_idx.keys())), bytes):
                        self.word_to_idx = {k.decode('latin1'): v for k, v in self.word_to_idx.items()}
                except Exception as e3:
                    # If all loading attempts fail, create a minimal vocabulary
                    print("Failed to load vocabulary, creating minimal vocabulary")
                    self.word_to_idx = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4}
        
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        
        # For BERT vocabulary, use BERT's special tokens
        # For custom vocabulary, try both BERT-style and custom style tokens
        if '[PAD]' in self.word_to_idx:
            self.pad_token_id = self.word_to_idx['[PAD]']
        elif '<PAD>' in self.word_to_idx:
            self.pad_token_id = self.word_to_idx['<PAD>']
        else:
            self.pad_token_id = 0
            
        if '[UNK]' in self.word_to_idx:
            self.unk_token_id = self.word_to_idx['[UNK]']
        elif '<UNK>' in self.word_to_idx:
            self.unk_token_id = self.word_to_idx['<UNK>']
        else:
            self.unk_token_id = 1
    
    def tokenize(self, text):
        """Tokenize text to list of words"""
        # Simple tokenization by splitting on whitespace and lowercasing
        return text.lower().split()
    
    def convert_tokens_to_ids(self, tokens):
        """Convert tokens to IDs"""
        return [self.word_to_idx.get(token, self.unk_token_id) for token in tokens]
    
    def __call__(self, text, truncation=True, max_length=512, padding='max_length', return_tensors=None):
        """Mimic the behavior of HuggingFace tokenizers"""
        if isinstance(text, str):
            tokens = self.tokenize(text)
            if truncation and len(tokens) > max_length:
                tokens = tokens[:max_length]
            
            ids = self.convert_tokens_to_ids(tokens)
            
            # Handle padding
            if padding == 'max_length':
                pad_length = max_length - len(ids)
                if pad_length > 0:
                    ids = ids + [self.pad_token_id] * pad_length
            
            if return_tensors == 'pt':
                return {'input_ids': torch.tensor([ids], dtype=torch.long)}
            else:
                return {'input_ids': [ids]}
        
        # Handle batch input
        elif isinstance(text, list):
            batch_ids = []
            for t in text:
                tokens = self.tokenize(t)
                if truncation and len(tokens) > max_length:
                    tokens = tokens[:max_length]
                ids = self.convert_tokens_to_ids(tokens)
                
                # Handle padding
                if padding == 'max_length':
                    pad_length = max_length - len(ids)
                    if pad_length > 0:
                        ids = ids + [self.pad_token_id] * pad_length
                
                batch_ids.append(ids)
            
            if return_tensors == 'pt':
                return {'input_ids': torch.tensor(batch_ids, dtype=torch.long)}
            else:
                return {'input_ids': batch_ids}

class FakeNewsDetector:
    """Helper class for easy inference"""
    def __init__(self, model_path, tokenizer_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.tokenizer = self.load_ensemble_model(model_path, tokenizer_path)
        self.model.to(self.device)
        self.model.eval()
    
    def load_ensemble_model(self, model_path, tokenizer_path):
        """Load the ensemble model from file"""
        # Load custom tokenizer
        tokenizer = CustomTokenizer(tokenizer_path)
        
        # Load the saved ensemble
        checkpoint = torch.load(model_path, map_location=self.device)
        params = checkpoint['params']
        
        # Recreate the individual models first
        models = []
        for i in range(5):  # Expecting 5 fold models
            model = AttentionBiLSTM(
                vocab_size=params["vocab_size"],
                embedding_dim=params["embedding_dim"],
                hidden_dim=params["hidden_dim"],
                output_dim=params["output_dim"],
                n_layers=params["n_layers"],
                dropout=params["dropout"],
                bidirectional=params["bidirectional"]
            ).to(self.device)
            models.append(model)
        
        # Create the ensemble
        ensemble = FakeNewsEnsembleModel(models)
        
        # Load state dict
        ensemble.load_state_dict(checkpoint['model_state_dict'])
        ensemble.eval()
        
        return ensemble, tokenizer
        
    def predict(self, text):
        """Predict if text is real or fake news"""
        # Tokenize input
        encoding = self.tokenizer(
            text, 
            truncation=True, 
            max_length=512, 
            padding='max_length', 
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        
        # Run prediction
        with torch.no_grad():
            prob, attention = self.model(input_ids)
            prob = prob.item()
            is_real = prob >= 0.5
            
        return {
            "is_real": bool(is_real),
            "prediction": "REAL" if is_real else "FAKE",
            "confidence": float(prob if is_real else (1 - prob)),
            "probability_real": float(prob)
        }

# Load the model once when the app starts
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          "../checkpoints", "best_model.pt")
TOKENIZER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "../data/processed", "vocab.pkl")
detector = None

try:
    print(f"Loading ensemble model from {MODEL_PATH}")
    print(f"Loading custom tokenizer from {TOKENIZER_PATH}")
    detector = FakeNewsDetector(MODEL_PATH, TOKENIZER_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html', title="Fake News Detector")

@app.route('/api/detect', methods=['POST'])
def detect_fake_news():
    """API endpoint for fake news detection"""
    if detector is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    
    try:
        result = detector.predict(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/detect', methods=['GET', 'POST'])
def detect_page():
    """Web page for fake news detection"""
    result = None
    
    if request.method == 'POST':
        text = request.form.get('text', '')
        if text and detector is not None:
            result = detector.predict(text)
    
    return render_template('detect.html', result=result)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for frontend prediction requests"""
    if detector is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    
    try:
        result = detector.predict(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Create templates directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"), exist_ok=True)

# Create a simple template if it doesn't exist
template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "detect.html")
if not os.path.exists(template_path):
    with open(template_path, "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detector</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        textarea { width: 100%; height: 200px; margin: 10px 0; }
        button { padding: 10px 20px; background: #4285f4; color: white; border: none; border-radius: 4px; }
        .result { margin-top: 20px; padding: 15px; border-radius: 4px; }
        .real { background-color: #d4edda; border: 1px solid #c3e6cb; }
        .fake { background-color: #f8d7da; border: 1px solid #f5c6cb; }
    </style>
</head>
<body>
    <h1>Fake News Detector</h1>
    <form method="post">
        <p>Enter news text to analyze:</p>
        <textarea name="text" required></textarea>
        <button type="submit">Detect</button>
    </form>
    
    {% if result %}
    <div class="result {{ 'real' if result.prediction == 'REAL' else 'fake' }}">
        <h2>Result: {{ result.prediction }}</h2>
        <p>Confidence: {{ "%.2f"|format(result.confidence*100) }}%</p>
        <p>Probability of being real news: {{ "%.2f"|format(result.probability_real*100) }}%</p>
    </div>
    {% endif %}
</body>
</html>
        """)

if __name__ == '__main__':
    app.run(debug=True) 