import os

import torch
from PIL import Image
from torchvision import transforms

from pistachio.src.models.lightning_train import TransferLearningModel

img_test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Provide the path to your trained model
model_path = "pistachio/models/transfer_learning_model.pth"


# Load the model
def load_model(model_path):
    model = TransferLearningModel()  # Instantiate your model class
    model.load_state_dict(torch.load(model_path))
    return model





def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_prediction(filename, model_path):
    model = load_model(model_path)
    classes, class_to_idx = find_classes("data/raw/")
    print("Available classes:", classes)

    img = Image.open(filename)
    img = img_test_transforms(img)
    img = img.unsqueeze(0)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model(img.to(device))
    #_, predicted = torch.max(outputs.data, 1)
    predicted = torch.round(torch.sigmoid(outputs.data)).int()

    predicted_class = [key for key, value in class_to_idx.items() if value == predicted.item()]
    print("Predicted class:", predicted_class)


# Provide the correct path to your example image
example_Kirmizi_path = 'data/raw/Kirmizi_Pistachio/kirmizi 110.jpg'  # Adjust the path accordingly
make_prediction(model, example_Kirmizi_path)

example_Siirt_path = 'data/raw/Siirt_Pistachio/siirt 110.jpg'  # Adjust the path accordingly
make_prediction(model, example_Siirt_path)
