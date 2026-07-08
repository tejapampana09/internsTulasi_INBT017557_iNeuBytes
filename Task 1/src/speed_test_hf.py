import os
import numpy as np
from datasets import load_dataset

print("Loading CIFAR-10 from Hugging Face...")
dataset = load_dataset("uoft-cs/cifar10")
print("Dataset keys:", dataset.keys())
print("Train size:", len(dataset['train']))
print("Test size:", len(dataset['test']))

# Test converting to numpy
first_img = np.array(dataset['train'][0]['img'])
first_label = dataset['train'][0]['label']
print("First image shape:", first_img.shape)
print("First label:", first_label)
