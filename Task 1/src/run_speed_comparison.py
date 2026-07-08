import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from datasets import load_dataset

# Load dataset (already cached, will be fast)
raw_dataset = load_dataset("uoft-cs/cifar10")
x_train_full = np.array([np.array(x) for x in raw_dataset['train']['img']], dtype=np.float32) / 255.0
y_train_full = np.array(raw_dataset['train']['label'], dtype=np.int32)

# Subset of 20000 images for speed test
x_train_sub = x_train_full[:20000]
y_train_sub = y_train_full[:20000]

print("Subset shapes:", x_train_sub.shape, y_train_sub.shape)

# Small Model (32 -> 64 -> 128)
model_small = models.Sequential([
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

model_small.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_small.summary()

print("Training small model for 1 epoch on 20000 images...")
start = time.time()
model_small.fit(x_train_sub, y_train_sub, epochs=1, batch_size=128, verbose=1)
elapsed_small = time.time() - start
print(f"Small model epoch took: {elapsed_small:.2f}s")
