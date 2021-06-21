import pandas as pd
import os
import os
import torch
import pandas as pd
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from hyperparameters import *
from utils import normalization_parameter
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder() # convert string into integer

# imgs = os.listdir(os.getcwd()+'/sample/img')
# df = pd.DataFrame(imgs)
#
# label_data = np.array(['dog','sink','ref','toilet','two man'])
# encoder.fit(label_data)
# encoded_label = encoder.transform(label_data)
# label = pd.DataFrame(encoded_label)
# res = pd.concat([df,label],axis=1)
# res.columns = ['images','labels']
# res.to_csv('sample/labels.csv')

"""
(1) 데이터셋 틀 만들기
"""
class CustomDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): csv 파일의 경로
            root_dir (string): 모든 이미지가 존재하는 디렉토리 경로
            transform (callable, optional): 샘플에 적용될 Optional transform
        """
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()

        img_names = os.path.join(self.root_dir,self.labels.iloc[idx,1])
        labels = self.labels.iloc[idx,2]

        image = io.imread(img_names)

        sample = {'image': image, 'image_name': img_names, 'label':labels}

        if self.transform:
            sample = {'image':self.transform(sample['image']), 'image_name': img_names, 'label':labels}

        return sample

imageNnameDataset = CustomDataset(csv_file='sample/labels.csv',root_dir='sample/img/')

"""
(2) transform 적용하기
"""

transformed_dataset = CustomDataset(csv_file='sample/labels.csv',root_dir='sample/img/',
                                    transform=transforms.Compose([transforms.ToPILImage(),
                                                                  transforms.ToTensor(),
                                                                  transforms.Resize(size=(args.img_size, args.img_size))
                                                                  ]))


for i in transformed_dataset:
    print(i)
"""
(3) DataLoader로 감싸기
"""
train_loader = DataLoader(transformed_dataset, batch_size=args.batch_size,shuffle=True)
mean, std = normalization_parameter(train_loader)





