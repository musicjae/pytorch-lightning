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


class SpatialGatingUnit(pl.LightningModule):
    def __init__(self, d_ffn, seq_len):
        super(SpatialGatingUnit,self).__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len,seq_len,kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias,1.0)

    def forward(self,x):
        u,v = x.chunk(2,dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        out = u*v
        return out


class gMLPBlock(pl.LightningModule):
    def __init__(self, d_model, d_ffn, seq_len):
        super(gMLPBlock,self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        out = x + residual
        return out


class gMLP(pl.LightningModule):
    def __init__(self, d_model=256, d_ffn=512, seq_len=256, num_layers=6):
        super(gMLP,self).__init__()
        self.model = nn.Sequential(
            *[gMLPBlock(d_model, d_ffn, seq_len) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.model(x)


def check_sizes(image_size, patch_size):
    sqrt_num_patches, remainder = divmod(image_size, patch_size)
    assert remainder == 0, "`image_size` must be divisibe by `patch_size`"
    num_patches = sqrt_num_patches ** 2
    return num_patches


class gMLPForImageClassification(gMLP):
    def __init__(
        self,
        image_size,
            lr,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        d_model=256,
        d_ffn=512,
        seq_len=256,
        num_layers=6,

    ):
        num_patches = check_sizes(image_size, patch_size)
        super().__init__(d_model, d_ffn, seq_len, num_layers)
        self.patcher = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )
        self.classifier = nn.Linear(d_model, num_classes)
        self.lr = lr

    def forward(self, x):
        patches = self.patcher(x)
        batch_size, num_channels, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_channels)
        embedding = self.model(patches)
        embedding = embedding.mean(dim=1)
        out = self.classifier(embedding)
        return out

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


class MLP(pl.LightningModule):
    def __init__(self,input_size, lr): # for auto_scale_batch_size
        super(MLP,self).__init__()
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

basic_mlp = MLP(input_size,lr)
gmlp = gMLPForImageClassification(input_size,lr)

