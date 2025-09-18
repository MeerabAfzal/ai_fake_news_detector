import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add parent directory to path to allow importing from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.utils import clean_text, spacy_tokenize, build_vocabulary, texts_to_sequences, pad_sequences

def prepare_data(max_vocab_size=15000, max_seq_length=100, test_size=0.2, random_state=42):
    """
    Prepare the dataset for training
    
    Args:
        max_vocab_size: Maximum size of vocabulary
        max_seq_length: Maximum sequence length for padding
        test_size: Proportion of the dataset to include in the test split
        random_state: Random seed for reproducibility
        
    Returns:
        None (saves processed data to files)
    """
    print("Loading data...")
    
    # Load fake news
    fake_df = pd.read_csv('data/Fake.csv')
    fake_df['label'] = 0  # 0 for fake
    
    # Load real news
    real_df = pd.read_csv('data/True.csv')
    real_df['label'] = 1  # 1 for real
    
    # Combine datasets
    df = pd.concat([fake_df, real_df], ignore_index=True)
    
    # Use title and text columns as input features
    # Check if both columns exist
    if 'title' in df.columns and 'text' in df.columns:
        print(f"Using title and text columns. Found {len(df)} articles.")
        # Combine title and text
        df['content'] = df['title'] + ' ' + df['text']
    elif 'text' in df.columns:
        print(f"Using only text column. Found {len(df)} articles.")
        df['content'] = df['text']
    else:
        raise ValueError("Dataset does not contain expected columns (text or title)")
    
    # Clean the text data
    print("Cleaning text data...")
    df['clean_content'] = df['content'].apply(clean_text)
    
    # Tokenize text
    print("Tokenizing text...")
    df['tokens'] = df['clean_content'].apply(lambda x: spacy_tokenize(x, remove_stopwords=True))
    
    # Build vocabulary
    print(f"Building vocabulary (max size: {max_vocab_size})...")
    vocab = build_vocabulary(df['tokens'].tolist(), max_vocab_size=max_vocab_size)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Convert texts to sequences
    print("Converting texts to sequences...")
    sequences = texts_to_sequences(df['tokens'].tolist(), vocab)
    
    # Pad sequences
    print(f"Padding sequences to length {max_seq_length}...")
    padded_sequences = pad_sequences(sequences, max_length=max_seq_length)
    
    # Split data into train and test sets
    print(f"Splitting data into train and test sets (test size: {test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, 
        df['label'].values, 
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']  # Ensure balanced classes in both splits
    )
    
    # Create directories if they don't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Save processed data
    print("Saving processed data...")
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_test.npy', y_test)
    
    # Save vocabulary and other metadata
    with open('data/processed/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
        
    with open('data/processed/metadata.pkl', 'wb') as f:
        metadata = {
            'max_seq_length': max_seq_length,
            'vocab_size': len(vocab),
            'class_distribution': {
                'train': {'fake': (y_train == 0).sum(), 'real': (y_train == 1).sum()},
                'test': {'fake': (y_test == 0).sum(), 'real': (y_test == 1).sum()}
            },
            'dataset_stats': {
                'total_articles': len(df),
                'fake_articles': (df['label'] == 0).sum(),
                'real_articles': (df['label'] == 1).sum()
            }
        }
        pickle.dump(metadata, f)
    
    print("Data preparation complete!")
    print(f"Saved {len(X_train)} training samples and {len(X_test)} test samples.")
    
    # Print class distribution
    print("\nClass distribution:")
    print(f"Train: {(y_train == 0).sum()} fake, {(y_train == 1).sum()} real")
    print(f"Test: {(y_test == 0).sum()} fake, {(y_test == 1).sum()} real")
    
    return None

if __name__ == "__main__":
    prepare_data(max_vocab_size=15000, max_seq_length=100) 