import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import AlexNet
from utils import *

## HyperParameters
lr = 1e-3
batch_size = 64
num_epoch = 10

ckpt_dir = './checkpoint'
log_dir = './log'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Augmentation
# Horizontal Reflection, altering intensities of RGB channels, and normalizing the image
transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ColorJitter(), transforms.RandomRotation(10), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_val = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 흐으으으으음..... 이미지넷 불러와야하는데 내가 학습을 시킬수가 없네... 돈이 부족하다...
dataset = datasets.MNIST(download=True, root='./', train=True, transform=transform_train)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

num_data = len(train_loader.dataset)
num_batch = np.ceil(num_data / batch_size)

## Model
alexNet = AlexNet().to(device)
params = alexNet.parameters()

fn_loss = nn.CrossEntropyLoss().to(device)
fn_pred = lambda output: torch.softmax(output, dim=1)
fn_acc = lambda pred, label: ((pred.max(dim=1)[1] == label).type(torch.float)).mean()

optim = torch.optim.Adam(params, lr=lr)

writer = SummaryWriter(log_dir=log_dir)

for epoch in range(1, num_epoch + 1):
    alexNet.train()

    loss_arr = []
    acc_arr = []

    for batch, (input, label) in enumerate(train_loader, 1):
        input = input.to(device)
        label = label.to(device)

        output = alexNet(input)
        pred = fn_pred(output)

        optim.zero_grad()

        loss = fn_loss(output, label)
        acc = fn_acc(pred, label)

        loss.backward()

        optim.step()

        loss_arr += [loss.item()]
        acc_arr += [acc.item()]

        print('TRAIN: EPOCH %04d/%04d | BATCH %04d/%04d | LOSS: %.4f | ACC %.4f' %
            (epoch, num_epoch, batch, num_batch, np.mean(loss_arr), np.mean(acc_arr)))
    
    writer.add_scalar('loss', np.mean(loss_arr), epoch)
    writer.add_scalar('acc', np.mean(acc_arr), epoch)

    save(ckpt_dir=ckpt_dir, net=alexNet, optim=optim, epoch=epoch)

writer.close()
