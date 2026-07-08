import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from datasets import load_dataset

# Set random seed
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("Loading dataset from Hugging Face...")
raw_dataset = load_dataset("uoft-cs/cifar10")

print("Preparing numpy arrays...")
x_train_full = np.array([np.array(x) for x in raw_dataset['train']['img']], dtype=np.float32) / 255.0
y_train_full = np.array(raw_dataset['train']['label'], dtype=np.int32)

x_test = np.array([np.array(x) for x in raw_dataset['test']['img']], dtype=np.float32) / 255.0
y_test = np.array(raw_dataset['test']['label'], dtype=np.int32)

# Create validation split (40000 train, 10000 validation)
indices = np.arange(len(x_train_full))
np.random.seed(SEED)
np.random.shuffle(indices)

train_idx = indices[:40000]
val_idx = indices[40000:]

x_train = x_train_full[train_idx]
y_train = y_train_full[train_idx]

x_val = x_train_full[val_idx]
y_val = y_train_full[val_idx]

print(f"Split sizes: Train={x_train.shape[0]}, Val={x_val.shape[0]}, Test={x_test.shape[0]}")

# Small Model (32 -> 64 -> 128)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("Training for 10 epochs...")
start = time.time()
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=128, verbose=1)
elapsed = time.time() - start
print(f"10 epochs training completed in {elapsed:.2f} seconds.")

# Evaluate on test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")
