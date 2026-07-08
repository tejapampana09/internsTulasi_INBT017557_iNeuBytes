import tensorflow as tf
from tensorflow.keras import layers, models
print("TensorFlow Version:", tf.__version__)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(10)
])
print("Model built successfully!")
print("Parameters:", model.count_params())
