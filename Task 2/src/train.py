import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns

def evaluate_metrics(y_true, y_pred):
    """
    Computes accuracy, macro precision, recall, and F1-score.
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1)
    }

def plot_learning_curves(history, filepath):
    """
    Plots training vs. validation accuracy and loss curves.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss curves
    ax1.plot(history.history['loss'], label='Train Loss', color='#e74c3c', lw=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', color='#2980b9', lw=2)
    ax1.set_title('Training vs. Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Accuracy curves
    ax2.plot(history.history['accuracy'], label='Train Accuracy', color='#2ecc71', lw=2)
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy', color='#9b59b6', lw=2)
    ax2.set_title('Training vs. Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

def plot_confusion_matrix_plot(y_true, y_pred, class_names, filepath):
    """
    Plots a heatmap for the confusion matrix.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

def download_glove():
    """
    Downloads the 50d GloVe file from Hugging Face if not already present locally.
    """
    local_path = "glove.6B.50d.txt"
    url = "https://huggingface.co/JeremiahZ/glove/resolve/main/glove.6B.50d.txt"
    
    if os.path.exists(local_path):
        print("GloVe embedding file already exists locally.")
        return local_path
        
    print(f"Downloading GloVe embeddings from {url}...")
    try:
        urllib.request.urlretrieve(url, local_path)
        print("Download complete.")
        return local_path
    except Exception as e:
        print(f"Failed to download GloVe file: {e}")
        return None

def load_glove_embeddings(word_index, vocab_size, embedding_dim=50):
    """
    Loads GloVe embeddings and creates an embedding matrix matching the word_index.
    """
    glove_path = download_glove()
    if not glove_path:
        print("Pre-trained embedding download failed. Falling back to Trainable Embedding.")
        return None
        
    print("Loading GloVe vectors into memory...")
    embeddings_index = {}
    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                # Some lines might have spacing issues, skip them
                if len(values) < embedding_dim + 1:
                    continue
                word = values[0]
                coefs = np.asarray(values[1:embedding_dim+1], dtype='float32')
                embeddings_index[word] = coefs
    except Exception as e:
        print(f"Error reading GloVe file: {e}")
        return None
        
    print(f"Found {len(embeddings_index)} word vectors in GloVe.")
    
    # Construct embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            # Words not found in embedding index will be initialized to random small values
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))
            
    return embedding_matrix

def perform_error_analysis(y_true, y_pred, raw_texts, num_examples=5):
    """
    Finds misclassified examples (sarcasm, negations, long context)
    and returns a list of dictionaries with review contents.
    """
    misclassified_indices = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    
    examples = []
    # Keywords to look for in misclassified text to identify sarcasm/negations/long context
    negation_words = {"not", "no", "never", "but", "however", "although", "except", "instead"}
    
    # Sort or search misclassified reviews
    for idx in misclassified_indices:
        text = raw_texts[idx]
        text_lower = text.lower()
        
        # Categorize
        category = "General"
        if any(w in text_lower for w in ["not good", "never liked", "wasn't", "didn't", "no good"]):
            category = "Negation"
        elif len(text.split()) > 200:
            category = "Long-Range Context"
        elif any(w in text_lower for w in ["sarcastic", "ironic", "great", "excellent", "love"]) and y_true[idx] == 0:
            # Positive words in a negative review often indicate sarcasm
            category = "Sarcasm / Irony"
            
        examples.append({
            'index': int(idx),
            'text': text,
            'true_label': int(y_true[idx]),
            'pred_label': int(y_pred[idx]),
            'category': category
        })
        
    # Return a diverse set of examples
    diverse_examples = []
    categories_found = set()
    
    # Try to get one of each category first
    for ex in examples:
        if ex['category'] not in categories_found:
            diverse_examples.append(ex)
            categories_found.add(ex['category'])
        if len(diverse_examples) >= num_examples:
            break
            
    # Fill remaining if needed
    if len(diverse_examples) < num_examples:
        for ex in examples:
            if ex not in diverse_examples:
                diverse_examples.append(ex)
            if len(diverse_examples) >= num_examples:
                break
                
    return diverse_examples
