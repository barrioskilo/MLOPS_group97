''''
# main.py
import click
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models.model import MyAwesomeModel

# this function loads data from data/processed/processed_data.pth and splits it into train and test sets and create data loaders for each set
# shuffle the data and 0.8 of the data is used for training and 0.2 for testing
def prepare_data():
    processed_data = torch.load('/Users/anderbarriocampos/Desktop/MLOPS_group97/data/processed/processed_data.pt')
    data = processed_data['data']
    labels = processed_data['labels']

    # Create train/test indices
    train_indices = torch.arange(0, 0.8*len(data))
    test_indices = torch.arange(0.8*len(data), len(data))

    # Convert indices to long type
    train_indices = train_indices.long()
    test_indices = test_indices.long()

    # Create datasets
    train_data = data[train_indices]
    train_labels = labels[train_indices]
    test_data = data[test_indices]
    test_labels = labels[test_indices]

    # Create data loaders
    train_loader = DataLoader(list(zip(train_data, train_labels)), batch_size=64, shuffle=True)
    test_loader = DataLoader(list(zip(test_data, test_labels)), batch_size=64, shuffle=True)

    return train_loader, test_loader



@click.group()
def cli():
    """Command line interface."""
    pass

@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=10, help="number of training epochs")
def train(lr, epochs):
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"Learning rate: {lr}")
    print(f"Number of epochs: {epochs}")

    # Load the data
    train_loader, _ = prepare_data()

    # Initialize model, loss function, and optimizer
    model = MyAwesomeModel()
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
    torch.save(model.state_dict(), 'models/pistachio_model.pt')

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    # Load the data
    _, test_loader = prepare_data()

    # Load the model
    model = MyAwesomeModel()
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

cli.add_command(train)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
'''

# main.py
import click
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models.model import MyAwesomeModel

# Function to prepare data
def prepare_data():
    processed_data = torch.load('/Users/anderbarriocampos/Desktop/MLOPS_group97/data/processed/processed_data.pt')
    data = processed_data['data']
    labels = processed_data['labels']

    # Create train/test indices
    split_index = int(0.8 * len(data))
    train_data, train_labels = data[:split_index], labels[:split_index]
    test_data, test_labels = data[split_index:], labels[split_index:]

    # Create data loaders
    train_loader = DataLoader(list(zip(train_data, train_labels)), batch_size=64, shuffle=True)
    test_loader = DataLoader(list(zip(test_data, test_labels)), batch_size=64, shuffle=True)

    return train_loader, test_loader

@click.group()
def cli():
    """Command line interface."""
    pass

@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=10, help="number of training epochs")
def train(lr, epochs):
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"Learning rate: {lr}")
    print(f"Number of epochs: {epochs}")

    # Load the data
    train_loader, _ = prepare_data()

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
    torch.save(model.state_dict(), 'models/pistachio_model.pt')

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

cli.add_command(train)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()


