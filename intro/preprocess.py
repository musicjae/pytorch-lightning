import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms as tt
from torchvision.datasets import ImageFolder

dspath = '../data/image-clf/dataset'
classes = os.listdir(dspath+'/training_set')

img_size= 64

train_tf = tt.Compose([tt.Resize((img_size,img_size)),
                       tt.RandomHorizontalFlip(),
                       tt.ToTensor()])

test_tf = tt.Compose([tt.Resize((img_size,img_size)),
                       tt.ToTensor()])

train_ds = ImageFolder(root=dspath+'/training_set',transform=train_tf)
val_ds = ImageFolder(root=dspath+'/test_set',transform=test_tf)

train_loader = DataLoader(train_ds, shuffle=True, batch_size=24)
test_loader = DataLoader(val_ds,batch_size=8)

