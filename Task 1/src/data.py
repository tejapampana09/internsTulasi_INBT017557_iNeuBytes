import numpy as np
import tensorflow as tf
from datasets import load_dataset

# Fixed random seed
SEED = 42

def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_and_preprocess_data():
    """
    Loads CIFAR-10 from Hugging Face, normalizes it, and splits it into
    train (40,000), validation (10,000), and test (10,000) splits.
    Reuses the same fixed random split for reproducibility.
    """
    set_seed(SEED)
    
    print("Loading CIFAR-10 from Hugging Face...")
    raw_dataset = load_dataset("uoft-cs/cifar10")
    
    print("Converting images to numpy arrays...")
    x_train_full = np.array([np.array(x) for x in raw_dataset['train']['img']], dtype=np.float32) / 255.0
    y_train_full = np.array(raw_dataset['train']['label'], dtype=np.int32)
    
    x_test = np.array([np.array(x) for x in raw_dataset['test']['img']], dtype=np.float32) / 255.0
    y_test = np.array(raw_dataset['test']['label'], dtype=np.int32)
    
    # Shuffle and split train/val
    indices = np.arange(len(x_train_full))
    # We use a local RandomState to ensure the split is completely independent of other global numpy calls
    rng = np.random.RandomState(SEED)
    rng.shuffle(indices)
    
    train_idx = indices[:10000]
    val_idx = indices[10000:20000]
    
    x_train = x_train_full[train_idx]
    y_train = y_train_full[train_idx]
    
    x_val = x_train_full[val_idx]
    y_val = y_train_full[val_idx]
    
    print(f"Data Loaded: Train={x_train.shape[0]}, Val={x_val.shape[0]}, Test={x_test.shape[0]}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
