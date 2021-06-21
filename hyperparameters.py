import argparse
import torch

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:',device)
parser = argparse.ArgumentParser(description='hyperparameters.....')
args = parser.parse_args("")

# =========== Training ============ #

args.batch_size = 2
args.img_size = 100
args.epochs = 200
args.lr = 1e-4
args.eps = 1e-8
args.total_steps = 1055 * args.epochs  # 총 훈련 스텝 =  배치반복 횟수 * 에폭 where 1055 is len(train_loader)