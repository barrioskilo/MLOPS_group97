import torch
from pistachio.src.models.lightning_train import TransferLearningModel

def test_output_shape():
    # Instantiate your model
    model = TransferLearningModel()

    # Define an example input with shape X
    example_input = torch.randn((1, 3, 600, 600))  # Adjust the shape according to your model's input requirements

    # Get the output from the model
    output = model(example_input)

    # Check if the output has the correct shape
    assert len(output.shape) == 2, "Output should be a 2D tensor"
    print("Test passed: output has the correct shape.")



