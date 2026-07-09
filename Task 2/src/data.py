import re
import numpy as np
import tensorflow as tf
from datasets import load_dataset

SEED = 42

# Inline standard list of English stopwords for offline-safety
STOPWORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", 
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
])

def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def clean_text(text):
    # Remove HTML tags (e.g. <br />)
    text = re.sub(r'<[^>]*>', ' ', text)
    # Remove punctuation and special characters, keep only words and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Lowercase and split
    words = text.lower().split()
    # Remove stopwords
    filtered_words = [w for w in words if w not in STOPWORDS]
    # Return cleaned sentence
    return " ".join(filtered_words)

def load_and_preprocess_data(train_size=5000, val_size=2500, test_size=2500):
    """
    Loads IMDb dataset, cleans text, and partitions it into seed-locked splits
    with equal class balance (positive/negative reviews).
    """
    set_seed(SEED)
    
    print("Loading IMDb dataset from Hugging Face...")
    dataset = load_dataset("imdb")
    
    # Extract train and test pools
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']
    
    # Class distribution analysis on full dataset
    unique, counts = np.unique(train_labels, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"Original train class distribution: {class_dist}")
    
    # Clean all texts
    print("Cleaning dataset texts...")
    cleaned_train_texts = [clean_text(t) for t in train_texts]
    cleaned_test_texts = [clean_text(t) for t in test_texts]
    
    # Partition seed-locked splits with perfect balance
    rng = np.random.RandomState(SEED)
    
    # For train/val, we split raw train data
    pos_train_indices = [i for i, l in enumerate(train_labels) if l == 1]
    neg_train_indices = [i for i, l in enumerate(train_labels) if l == 0]
    
    rng.shuffle(pos_train_indices)
    rng.shuffle(neg_train_indices)
    
    # Train split: half positive, half negative
    n_train_half = train_size // 2
    n_val_half = val_size // 2
    
    train_idx = pos_train_indices[:n_train_half] + neg_train_indices[:n_train_half]
    val_idx = pos_train_indices[n_train_half:n_train_half + n_val_half] + neg_train_indices[n_train_half:n_train_half + n_val_half]
    
    # For test split
    pos_test_indices = [i for i, l in enumerate(test_labels) if l == 1]
    neg_test_indices = [i for i, l in enumerate(test_labels) if l == 0]
    
    rng.shuffle(pos_test_indices)
    rng.shuffle(neg_test_indices)
    
    n_test_half = test_size // 2
    test_idx = pos_test_indices[:n_test_half] + neg_test_indices[:n_test_half]
    
    # Shuffle indices to mix classes
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    
    x_train = [cleaned_train_texts[i] for i in train_idx]
    y_train = np.array([train_labels[i] for i in train_idx])
    
    x_val = [cleaned_train_texts[i] for i in val_idx]
    y_val = np.array([train_labels[i] for i in val_idx])
    
    x_test = [cleaned_test_texts[i] for i in test_idx]
    y_test = np.array([test_labels[i] for i in test_idx])
    
    # Also save raw (uncleaned) versions of test for negation/sarcasm error analysis
    raw_test_texts = [test_texts[i] for i in test_idx]
    
    print(f"Data Partitioned: Train={len(x_train)}, Val={len(x_val)}, Test={len(x_test)}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), raw_test_texts
