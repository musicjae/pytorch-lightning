import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

import pytorch_lightning as pl

class dataModule(pl.LightningModule):

    def __init__(self,data_dir:str='../data/image-clf/dataset',batch_size=16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))
        ])

        self.dims = (1,28,28) # image shape
        self.num_classes=2

    def prepare_data(self):
        self.train_ds = ImageFolder(root=self.data_dir + '/training_set', transform=self.transform)
        self.test_ds = ImageFolder(root=self.data_dir + '/test_set', transform=self.transform)

    def setup(self,stage=None):

        if stage=='fit' or stage is None:
            self.train,self.val = random_split(self.train_ds,[7000,1000])
        if stage =='test' or stage is None:
            self.test = self.test_ds

    def train_loader(self):
        return DataLoader(self.train,batch_size=self.batch_size)

    def val_loader(self):
        return DataLoader(self.val,batch_size=self.batch_size)

    def test_loader(self):
        return DataLoader(self.test,batch_size=self.batch_size)



