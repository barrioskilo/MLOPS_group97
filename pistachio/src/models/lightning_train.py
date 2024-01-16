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
import os

import ssl

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_default_https_context 
ssl._create_default_https_context = ssl._create_unverified_context



class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # Log the images as wandb Image
        trainer.logger.experiment.log({
            "examples": [wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                         for x, pred, y in zip(val_imgs[:self.num_samples],
                                               preds[:self.num_samples],
                                               val_labels[:self.num_samples])]
        })

class MyAwesomeModel(pl.LightningModule):
    def __init__(self, num_classes=2):
        super().__init__()

        # Load SqueezeNet model
        '''
        self.resnet18_net = models.resnet18(pretrained=True)
        in_features = self.resnet18_net.fc.in_features
        self.resnet18_net.fc = nn.Linear(in_features, num_classes) 
        '''

        self.resnet18_net = models.resnet18(pretrained=True)
        for param in self.resnet18_net.parameters():
            param.requires_grad = False

        self.resnet18_net = self.resnet18_net.to(device=self.device)

        self.learning_rate = 0.001
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='binary')

    def forward(self, x):
        return self.resnet18_net(x)

    def training_step(self, batch, batch_idx):
        
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.resnet18_net.fc.parameters(), lr=self.learning_rate)
        return optimizer
    

class TransferLearningModel(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "resnet50",
        train_bn: bool = False,
        milestones: tuple = (2, 4),
        batch_size: int = 32,
        lr: float = 1e-3,
        lr_scheduler_gamma: float = 1e-1,
        num_workers: int = 6,
        **kwargs,
    ) -> None:
        """TransferLearningModel.
        Args:
            backbone: Name (as in ``torchvision.models``) of the feature extractor
            train_bn: Whether the BatchNorm layers should be trainable
            milestones: List of two epochs milestones
            lr: Initial learning rate
            lr_scheduler_gamma: Factor by which the learning rate is reduced at each milestone
        """
        super().__init__()
        self.backbone = backbone
        self.train_bn = train_bn
        self.milestones = milestones
        self.batch_size = batch_size
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers
        self.__build_model()
        self.train_acc = Accuracy(task='binary')
        self.valid_acc = Accuracy(task='binary')
        self.save_hyperparameters()
    def __build_model(self):
        model_func = getattr(models, self.backbone)
        backbone = model_func(pretrained=True)
        _layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*_layers)
        _fc_layers = [nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 32), nn.Linear(32, 1)]
        self.fc = nn.Sequential(*_fc_layers)
        self.loss_func = F.binary_cross_entropy_with_logits
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.squeeze(-1).squeeze(-1)   
        x = self.fc(x)
        return x
    def loss(self, logits, labels):
            return self.loss_func(input=logits, target=labels)
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self.forward(x)
        y_scores = torch.sigmoid(y_logits)
        y_true = y.view((-1, 1)).type_as(x)
        train_loss = self.loss(y_logits, y_true)
        self.log("train_acc", self.train_acc(y_scores, y_true.int()), prog_bar=True)
        return train_loss
    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        y_scores = torch.sigmoid(y_logits)
        y_true = y.view((-1, 1)).type_as(x)
# 2. Compute loss
        self.log("val_loss", self.loss(y_logits, y_true), prog_bar=True)
# 3. Compute accuracy:
        self.log("val_acc", self.valid_acc(y_scores, y_true.int()), prog_bar=True)
    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        optimizer = torch.optim.Adam(trainable_parameters, lr=self.lr)
        scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_scheduler_gamma)
        return [optimizer], [scheduler]


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def train(input_filepath):
    """Main function."""
    # Load the data
    dm = PistachioDataModule(input_filepath, batch_size=32)
    dm.setup()

    # Set up WandB logger with SSL verification disabled
    wandb_logger = WandbLogger(project='wandb-lightning', job_type='train')
    os.environ["WANDB_IGNORE_GLOBS"] = "*.pem"  # This line disables SSL verification

    model = TransferLearningModel()
    trainer = pl.Trainer(max_epochs=3, logger=wandb_logger)
    trainer.fit(model=model, datamodule=dm)

if __name__ == "__main__":
    train()

