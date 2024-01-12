# tests/test_model.py
import pytest
import torch
from pistachio.models.model import MyAwesomeModel  # Import your model class here

def test_error_on_wrong_shape():
    # Instantiate your model
    model = MyAwesomeModel()

    # Create input tensor with incorrect shape
    wrong_shape_input = torch.randn(1, 2, 3)  # Assuming the expected shape is [3, 600, 600]

    # Check if ValueError is raised with the expected message
    with pytest.raises(ValueError, match='Expected input to be a 4D tensor'):
        model(wrong_shape_input)
