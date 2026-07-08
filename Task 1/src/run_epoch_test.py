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
# Convert images and labels to numpy arrays
x_train_full = np.array([np.array(x) for x in raw_dataset['train']['img']], dtype=np.float32) / 255.0
y_train_full = np.array(raw_dataset['train']['label'], dtype=np.int32)

x_test = np.array([np.array(x) for x in raw_dataset['test']['img']], dtype=np.float32) / 255.0
y_test = np.array(raw_dataset['test']['label'], dtype=np.int32)

print("Full train shape:", x_train_full.shape)
print("Test shape:", x_test.shape)

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

# Baseline AlexNet-style model adapted for 32x32 images
# Requirements: "Build an AlexNet-style CNN adapted for small (32x32) images.
# Use convolutional layers with increasing filters (e.g. 64 -> 128 -> 256), ReLU activations, pooling layers, and fully connected layers for classification.
# Keep this a clean baseline - no data augmentation, no batch normalization, no dropout"
model = models.Sequential([
    # Conv 1
    layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    # Conv 2
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # Conv 3
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # Flatten
    layers.Flatten(),
    # FC 1
    layers.Dense(256, activation='relu'),
    # FC 2
    layers.Dense(128, activation='relu'),
    # Output
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("Training for 1 epoch...")
start = time.time()
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1, batch_size=128)
elapsed = time.time() - start
print(f"1 epoch training completed in {elapsed:.2f} seconds.")
