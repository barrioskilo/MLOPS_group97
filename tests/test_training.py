import torch
from pistachio.models.model import MyAwesomeModel  # Import your model class here

def test_model_forward_pass():
    # Instantiate your model
    model = MyAwesomeModel()

    # Define a sample input tensor (adjust based on your model's input requirements)
    sample_input = torch.randn((1, 3, 600, 600))

    # Forward pass
    output = model(sample_input)

    # Define the expected output shape (adjust based on your model's architecture)
    expected_output_shape = (1, 2)  # Example: a binary classification model with 2 output classes

    # Check if the output shape matches the expected shape
    assert output.shape == expected_output_shape, f"Output shape does not match the expected shape. Expected {expected_output_shape}, got {output.shape}"

    print("Model forward pass test passed.")

# Uncomment the line below if you want to run the test immediately
# test_model_forward_pass()

