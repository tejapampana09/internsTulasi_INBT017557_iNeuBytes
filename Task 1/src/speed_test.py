import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import time
import tensorflow as tf
from tensorflow.keras import layers, models, datasets

print("TensorFlow Version:", tf.__version__)
print("Downloading/Loading CIFAR-10...")
start_time = time.time()
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
print(f"Loaded in {time.time() - start_time:.2f}s")
print("x_train shape:", x_train.shape)

# Normalize
x_train, x_test = x_train / 255.0, x_test / 255.0

# Take a subset of 10000 images for speed test
x_train_sub = x_train[:1000]
y_train_sub = y_train[:1000]

# Simple model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("Starting training of 1 epoch on 1000 samples...")
start_time = time.time()
model.fit(x_train_sub, y_train_sub, epochs=1, batch_size=64, verbose=1)
elapsed = time.time() - start_time
print(f"1 epoch on 1000 samples took: {elapsed:.2f}s")
print(f"Extrapolated time for 50000 samples (full epoch): {elapsed * 50:.2f}s")
