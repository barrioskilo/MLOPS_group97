import torch
from pistachio.models.model import MyAwesomeModel  # Import your model class here

def test_output_shape():
    # Instantiate your model
    model = MyAwesomeModel()

    # Define an example input with shape X
    example_input = torch.randn((1, 3, 600, 600))  # Adjust the shape according to your model's input requirements

    # Get the output from the model
    output = model(example_input)

    # Define the expected output shape Y
    expected_output_shape = (1, 2)  # Adjust the shape according to your model's output

    # Check if the output shape matches the expected shape
    assert output.shape == expected_output_shape, "Output shape does not match the expected shape."
    print("Test passed: output shape matches the expected shape.")

