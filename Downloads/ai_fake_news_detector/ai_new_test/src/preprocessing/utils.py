import re
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Please run: python -m spacy download en_core_web_sm")

def clean_text(text):
    """
    Clean and preprocess text data
    """
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Expand contractions (simple version)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)
    
    return text

def tokenize_text(text, remove_stopwords=True):
    """
    Tokenize text using NLTK
    """
    if not text:
        return []
        
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

def spacy_tokenize(text, remove_stopwords=True):
    """
    Tokenize text using spaCy (more advanced)
    """
    if not isinstance(text, str) or not text:
        return []
    
    doc = nlp(text)
    
    if remove_stopwords:
        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    else:
        tokens = [token.lemma_ for token in doc if token.is_alpha]
    
    return tokens

def build_vocabulary(tokenized_texts, max_vocab_size=15000):
    """
    Build vocabulary from tokenized texts
    """
    word_freq = {}
    
    for text in tokenized_texts:
        for token in text:
            if token in word_freq:
                word_freq[token] += 1
            else:
                word_freq[token] = 1
    
    # Sort by frequency and limit size
    sorted_vocab = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Add special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1}
    
    # Add most frequent words
    for i, (word, _) in enumerate(sorted_vocab[:max_vocab_size-2]):
        vocab[word] = i + 2
    
    return vocab

def texts_to_sequences(tokenized_texts, vocab):
    """
    Convert tokenized texts to sequences of indices
    """
    sequences = []
    for text in tokenized_texts:
        sequence = [vocab.get(token, vocab['<UNK>']) for token in text]
        sequences.append(sequence)
    
    return sequences

def pad_sequences(sequences, max_length=None, padding='post'):
    """
    Pad sequences to the same length
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded_sequences = []
    for seq in sequences:
        if len(seq) > max_length:
            # Truncate
            padded_seq = seq[:max_length]
        else:
            # Pad
            padded_seq = seq + [0] * (max_length - len(seq)) if padding == 'post' else [0] * (max_length - len(seq)) + seq
        
        padded_sequences.append(padded_seq)
    
    return np.array(padded_sequences) 