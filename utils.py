import numpy as np
import tqdm

def normalization_parameter(dataloader):
    mean = 0.
    std = 0.
    nb_samples = len(dataloader.dataset)
    # tqdm은 진행상태를 알려주는 함수
    for _,data in enumerate(dataloader):
        batch_samples = data['image'].size(0)
        data = data['image'].view(batch_samples, data['image'].size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
    mean /= nb_samples
    std /= nb_samples
    return mean.numpy(),std.numpy()