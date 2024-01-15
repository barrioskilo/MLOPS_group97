import torch
from pistachio.models.model import MyAwesomeModel

def test_forward_pass_coverage():
    # Instantiate your model
    model = MyAwesomeModel()

    # Create a dummy input tensor
    dummy_input = torch.randn((1, 3, 600, 600))

    # Perform a forward pass
    output = model(dummy_input)

    # Add assertions to check the output shape or other properties
    assert output.shape == (1, 2), "Output shape should be (1, 2)"



