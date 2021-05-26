import pytorch_lightning as pl
from torchvision.datasets import ImageFolder
from pytorch_lightning.metrics.functional import accuracy
import torch
from pytorch_lightning import Trainer, tuner
from torch.nn import functional as F
from preprocess import train_loader,test_loader


class MNISTModel(pl.LightningModule):

    def __init__(self): # for auto_scale_batch_size
        super(MNISTModel,self).__init__()
        self.l1 = torch.nn.Linear(64*64*3,2)

    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.l1(x)
        x = torch.relu(x)
        return x

    def training_step(self,batch,batch_nb):
        x,y = batch
        loss = F.cross_entropy(self(x),y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.learning_rate)

model = MNISTModel()

"""
By using the Trainer you automatically get:

Tensorboard logging
Model checkpointing
Training and validation loop
early-stopping
"""
trainer = pl.Trainer(max_epochs=3,progress_bar_refresh_rate=20, auto_scale_batch_size=True) # bach size auto finder
tuner = tuner.tuning.Tuner(trainer)
model.learning_rate
trainer.fit(model)
#trainer.fit(model,train_loader)
