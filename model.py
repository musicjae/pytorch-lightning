import pytorch_lightning as pl
from torchvision.datasets import ImageFolder
from pytorch_lightning.metrics.functional import accuracy
import torch
from pytorch_lightning import Trainer, tuner
from torch.nn import functional as F
import torch.nn as nn
from hyperparameters import *
from data import *

input_size = args.img_size
lr = args.lr

class MNISTModel(pl.LightningModule):

    def __init__(self,input_size, lr): # for auto_scale_batch_size
        super(MNISTModel,self).__init__()
        self.l1 = torch.nn.Linear(input_size*input_size*3,5)
        self.lr = lr

    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.l1(x)
        x = torch.relu(x)
        return x

    def cross_entropy_loss(self,logits,labels):
        loss = nn.CrossEntropyLoss()
        return loss(logits,labels)

    def training_step(self,batch,batch_nb):
        x,y = batch['image'], batch['label']
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits,y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.lr)

model = MNISTModel(input_size,lr)
print(model)
"""
By using the Trainer you automatically get:
Tensorboard logging
Model checkpointing
Training and validation loop
early-stopping
"""
trainer = pl.Trainer(max_epochs=20,progress_bar_refresh_rate=20, auto_scale_batch_size=True) # bach size auto finder
tuner = tuner.tuning.Tuner(trainer)
trainer.fit(model,train_loader)