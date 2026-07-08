import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def train_model(model, data, epochs=10, batch_size=128, optimizer='adam', model_name='model', results_dir='results'):
    """
    Trains a model and returns training history, training time, and parameter count.
    """
    (x_train, y_train), (x_val, y_val) = data
    
    # Create results folder if not exists
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    if not model.built:
        model.build((None, 32, 32, 3))
    param_count = model.count_params()
    print(f"\n--- Training {model_name} ({param_count:,} parameters) ---")
    
    # Fit model and measure time
    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"Training finished in {training_time:.2f} seconds.")
    
    # Save model weights
    model_path = os.path.join(results_dir, "models", f"{model_name}.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return history.history, training_time, param_count

def evaluate_model(model, x_test, y_test):
    """
    Evaluates the model on test data.
    Returns: accuracy, loss, precision, recall, f1_score, and predicted labels.
    """
    # Evaluate raw loss and accuracy
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    # Get predictions
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate precision, recall, and f1 (macro average)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }, y_pred

def plot_curves(history, model_name, save_dir='plots'):
    """
    Plots training and validation accuracy and loss curves.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(history['accuracy']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['accuracy'], 'bo-', label='Training Acc')
    plt.plot(epochs, history['val_accuracy'], 'ro-', label='Validation Acc')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"{model_name}_learning_curves.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Learning curves saved to {plot_path}")

def save_confusion_matrix_plot(y_true, y_pred, model_name, save_dir='plots/confusion_matrices'):
    """
    Computes and plots confusion matrix. Identifies and returns top 3 most confused pairs.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
    plt.yticks(tick_marks, CLASS_NAMES)
    
    # Label each cell
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
            
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Confusion matrix plot saved to {plot_path}")
    
    # Identify most confused pairs (exclude diagonal)
    confused_list = []
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i != j:
                confused_list.append((cm[i, j], CLASS_NAMES[i], CLASS_NAMES[j]))
                
    confused_list.sort(reverse=True, key=lambda x: x[0])
    
    # Return top 3 confused pairs
    return confused_list[:3]
