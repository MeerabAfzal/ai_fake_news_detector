import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import FakeNewsDetector

def main():
    """
    Check if GPU is available and provide instructions for training on Colab
    """
    print("Checking for GPU...")
    
    if torch.cuda.is_available():
        print("GPU is available! You can train locally.")
        # You could implement local training here if desired
    else:
        print("GPU not available locally. Redirecting to Google Colab for training.")
        print("\n" + "="*80)
        print("TRAINING INSTRUCTIONS FOR GOOGLE COLAB")
        print("="*80)
        print("""
1. Upload the following files to Google Colab:
   - training_on_colab.txt (rename to training_on_colab.py)
   - model.py (from src/model/)
   - Processed data files (from data/processed/):
     - X_train.npy
     - X_test.npy
     - y_train.npy
     - y_test.npy
     - vocab.pkl
     - metadata.pkl

2. Run the training script:
   ```python
   !python training_on_colab.py
   ```

3. After training, download:
   - All files in the 'checkpoints' directory, especially:
     - best_model.pt
     - training_history.pkl
     - training_history.png

4. Place the downloaded files in your local 'checkpoints' directory.
        """)
        print("="*80)
        
        # Create a dummy model architecture file for Colab upload
        os.makedirs("for_colab_upload", exist_ok=True)
        
        # Copy the model.py file to the upload directory
        import shutil
        shutil.copy(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "model.py"),
            os.path.join("for_colab_upload", "model.py")
        )
        
        # Copy the training script to the upload directory
        shutil.copy(
            "training_on_colab.txt",
            os.path.join("for_colab_upload", "training_on_colab.py")
        )
        
        print(f"\nFiles have been prepared in the 'for_colab_upload' directory for easy upload to Colab.")
        
if __name__ == "__main__":
    main() 