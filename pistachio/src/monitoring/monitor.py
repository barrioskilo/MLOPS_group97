import os
from PIL import Image
from torchvision import transforms
import torch
from pistachio.src.models.lightning_train import TransferLearningModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import wandb
from pistachio.src.data.make_lightning_dataset import PistachioDataModule
import click
from torchmetrics import Accuracy
from pytorch_lightning.loggers import WandbLogger
from torchvision import models
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from evidently.test_suite import TestSuite
from evidently.tests import *
import pandas as pd
import json


# Load the model
def load_model(model_path):
    model = TransferLearningModel()  # Instantiate your model class
    model.load_state_dict(torch.load(model_path))
    return model

input_filepath = 'data/raw/'
dm = PistachioDataModule(input_filepath, batch_size=32)
dm.setup()

# Provide the path to your trained model
model_path = 'pistachio/models/transfer_learning_model.pth'
model = load_model(model_path)

test_labels = np.array([])
test_preds = np.array([])
test_preds_corrupted = np.array([])
#test_labels = []
#test_preds = []
#test_preds_corrupted = []


with (torch.no_grad()):
    for batch in dm.test_dataloader():
        dataiter = iter(dm.test_dataloader())
        x, y = next(dataiter)
        #model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        y_pred = model(x.to(device)).squeeze()
        y_pred = torch.round(torch.sigmoid(y_pred)).int().numpy()
        #y_pred = torch.round(torch.sigmoid(y_pred)).int()
        x_corrupted = transforms.functional.rotate(x, 90)
        y_pred_corrupted = model(x_corrupted.to(device)).squeeze()
        y_pred_corrupted = torch.round(torch.sigmoid(y_pred_corrupted)).int().numpy()
        #y_pred_corrupted = torch.round(torch.sigmoid(y_pred_corrupted)).int()
        test_labels = np.append(test_labels, y.numpy())
        test_preds = np.append(test_preds, y_pred)
        test_preds_corrupted = np.append(test_preds_corrupted, y_pred_corrupted)


#test_labels = torch.cat(test_labels, dim=0)
#test_preds = torch.cat(test_preds, dim=0)
#test_labels = torch.cat(test_labels, dim=0)

print('Original data Accuracy: ', (test_preds == test_labels).mean())
print('Corrupted data Accuracy: ', (test_preds_corrupted == test_labels).mean())


#dataset-level tests
prob_classification_performance_dataset_tests = TestSuite(tests=[
    TestAccuracyScore(),
    TestPrecisionScore(),
    TestRecallScore(),
    TestF1Score(),
    #TestRocAuc(),
    #TestLogLoss(),
    TestPrecisionByClass(label=0),
    TestPrecisionByClass(label=1),
    TestRecallByClass(label=0),
    TestRecallByClass(label=1),
    TestF1ByClass(label=0),
    TestF1ByClass(label=1),

])

# Building Ref data
#target = pd.DataFrame(data={'target': test_labels.numpy()})
#prediction = pd.DataFrame(data={'prediction': test_preds.numpy()})
target = pd.DataFrame(data={'target': test_labels})
prediction = pd.DataFrame(data={'prediction': test_preds})
ref = pd.concat([target,prediction], axis=1)

# Building Cur data
#prediction_corrupted = pd.DataFrame(data={'prediction': test_preds_corrupted.numpy()})
prediction_corrupted = pd.DataFrame(data={'prediction': test_preds_corrupted})
cur = pd.concat([target,prediction_corrupted], axis=1)


prob_classification_performance_dataset_tests.run(reference_data=ref, current_data=cur)
result_dict = prob_classification_performance_dataset_tests.as_dict()
prob_classification_performance_dataset_tests.save_html('reports/monitoring.html')

# Specify the path to the JSON file
json_file_path = 'reports/result_dict.json'

# Save the dictionary to a JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(result_dict, json_file)

# Optionally, you can print a message indicating the successful save
print(f"Result dictionary saved to {json_file_path}")

