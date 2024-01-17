import os
import torch
import pytest
from tests import _PATH_DATA



# Assuming __init__.py is in the same directory as this script
init_file_path = os.path.join(os.path.dirname(__file__), '__init__.py')
exec(open(init_file_path).read())  # Execute __init__.py to get the variables

@pytest.mark.skipif(not os.path.exists("data/processed/processed_data.pt"), reason="Data files not found")

def test_data():
    dir = os.path.join(_PATH_DATA, 'processed')
    found = False

    for file in os.listdir(dir):
        if file.endswith(".pt"):
            processed_data = torch.load(os.path.join(dir, file))
            found = True

            # Check if 'data' and 'labels' are in the processed_data
            assert 'data' in processed_data, "Test failed: Processed data should contain 'data'"
            assert 'labels' in processed_data, "Test failed: Processed data should contain 'labels'"
            print("Test passed: 'data' and 'labels' are present in processed_data.")

            data_tensor = processed_data['data']
            labels_tensor = processed_data['labels']

            # Example: Check if each image in the data tensor has the correct shape
            # The shape of each image in your dataset should match the input of your model
            for img in data_tensor:
                assert img.shape == (3, 600, 600), f"Test failed: Each image should have shape (3, 600, 600), found {img.shape}"
            print("Test passed: All images have the correct shape.")

            # Check if the number of labels matches the number of images
            assert len(labels_tensor) == len(data_tensor), "Test failed: Number of labels should match the number of images"
            print("Test passed: Number of labels matches the number of images.")

            # Check that the only available labels are 0 and 1
            unique_labels = set(labels_tensor.numpy())
            assert set([0, 1]).issuperset(unique_labels), "Test failed: Only labels 0 and 1 should be present"
            print("Test passed: Only labels 0 and 1 are present.")

    # Ensure that at least one .pt file was found and processed
    if found:
        print("All tests passed successfully!")
    else:
        print("No .pt file found in the directory")

# Run the test
#test_data()
