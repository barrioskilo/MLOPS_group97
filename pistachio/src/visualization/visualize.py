import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
from pistachio.models.model import MyAwesomeModel

# Function to load preprocessed data
def load_processed_data(data_path):
    processed_data = torch.load(data_path)
    return processed_data['data'], processed_data['labels']

# Function to extract intermediate representations using a pre-trained model
def extract_intermediate_representations(model, images):
    model.eval()
    intermediate_features = model(images)
    return intermediate_features.detach().numpy()

# Function to visualize features using t-SNE
def visualize_features(features, labels):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)

    # Create a scatter plot with subplots for data distribution, intermediate features, and original features
    plt.figure(figsize=(18, 5))

    # Subplot for t-SNE visualization
    plt.subplot(1, 3, 1)
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis')
    plt.title('t-SNE Visualization of Intermediate Features')
    plt.colorbar()

    # Subplot for data distribution
    plt.subplot(1, 3, 2)
    plt.hist(labels, bins=len(np.unique(labels)), color='skyblue', edgecolor='black')
    plt.title('Data Distribution')
    plt.xlabel('Class Label')
    plt.ylabel('Count')

    # Subplot for histogram of intermediate features
    plt.subplot(1, 3, 3)
    plt.hist(features.flatten(), bins=50, color='orange', edgecolor='black')
    plt.title('Histogram of Intermediate Features')
    plt.xlabel('Feature Value')
    plt.ylabel('Count')

    plt.tight_layout()

    # Save the visualization
    plt.savefig('reports/figures/visualization.png')
    plt.show()

model_path = 'pistachio/models/pistachio_model.pth'
model = MyAwesomeModel()  # Instantiate your model
model.load_state_dict(torch.load(model_path))
model.eval()

processed_data_path = 'data/processed/processed_data.pt'
images, labels = load_processed_data(processed_data_path)

intermediate_features = extract_intermediate_representations(model, images)

visualize_features(intermediate_features, labels)