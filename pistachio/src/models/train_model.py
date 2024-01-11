# main.py
import click
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from pistachio.models.model import MyAwesomeModel

import hydra
from omegaconf import DictConfig, OmegaConf


# Function to prepare data
# Function to prepare shuffled data

def prepare_data(random_seed, portion):
    processed_data = torch.load('data/processed/processed_data.pt')
    data = processed_data['data']
    labels = processed_data['labels']

    # Shuffle data
    indices = list(range(len(data)))
    torch.manual_seed(random_seed)  # Set a seed for reproducibility
    torch.randperm(len(indices))
    data = [data[i] for i in indices]
    labels = [labels[i] for i in indices]

    # Create train/test indices
    split_index = int(portion * len(data))
    train_data, train_labels = data[:split_index], labels[:split_index]
    test_data, test_labels = data[split_index:], labels[split_index:]

    # Create data loaders
    train_loader = DataLoader(list(zip(train_data, train_labels)), batch_size=64, shuffle=True)
    test_loader = DataLoader(list(zip(test_data, test_labels)), batch_size=64, shuffle=True)

    return train_loader, test_loader

'''
@click.group()
def cli():
    """Command line interface."""
    pass

@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=10, help="number of training epochs")
'''

@hydra.main(version_base=None, config_path="conf", config_name="config")

def train(cfg : DictConfig):
    """Train a model on MNIST."""
    print("Training day and night")
    print(OmegaConf.to_yaml(cfg))
    lr = cfg.hyperparameters.learning_rate
    epochs = cfg.hyperparameters.epochs
    random_seed = cfg.data.random_seed
    portion = cfg.data.portion
    #print(f"Learning rate: {lr}")
    #print(f"Number of epochs: {epochs}")

    # Load the data
    train_loader, _ = prepare_data(random_seed, portion)

    # Initialize model, loss function, and optimizer
    model = MyAwesomeModel()  # Adjust input_size and hidden_size accordingly
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")

    # Save the trained model
    torch.save(model.state_dict(), 'pistachio/models/pistachio_model.pt')

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    # Load the data
    _, test_loader = prepare_data()

    # Load the model
    model = MyAwesomeModel()  # Adjust input_size and hidden_size accordingly
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy*100:.2f}%")

'''
cli.add_command(train)
cli.add_command(evaluate)
'''

if __name__ == "__main__":
    train()


