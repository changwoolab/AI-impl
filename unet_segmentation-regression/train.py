import argparse

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from torchvision import transforms, datasets

from model import UNet
from dataset import *
from util import *

## Arg Parser
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument('--batch_size', default=4, type=int, dest='batch_size')
parser.add_argument('--num_epoch', default=100, type=int, dest='num_epoch')

parser.add_argument('--data_dir', default='./datasets/images', type=str, dest='data_dir')
parser.add_argument('--ckpt_dir', default='./checkpoint', type=str, dest='ckpt_dir')
parser.add_argument('--log_dir', default='./log', type=str, dest='log_dir')
parser.add_argument('--result_dir', default='./result', type=str, dest='result_dir')
parser.add_argument('--mode', default='train', type=str, dest='mode')
parser.add_argument('--train_continue', default='off', type=str, dest='train_continue')

parser.add_argument('--task', default='denoising', choices=['denoising', 'inpainting', 'super_resolution'], type=str, dest='task')
parser.add_argument('--opts', nargs='+', default=['random', 30.0], dest='opts')

parser.add_argument('--ny', default=320, type=int, dest='ny')
parser.add_argument('--nx', default=480, type=int, dest='nx')
parser.add_argument('--nch', default=3, type=int, dest='nch')
parser.add_argument('--nker', default=64, type=int, dest='nker')

parser.add_argument('--network', default='unet', choices=['unet', 'resnet', 'autoencoder'], type=str, dest='network')
parser.add_argument('--learning_type', default='plain', choices=['plain', 'residual'], type=str, dest='learning_type')

args = parser.parse_args()

# 트레이닝 파라미터 설정
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir
mode = args.mode
train_continue = args.train_continue

task = args.task
opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float32)]

ny = args.ny
nx = args.nx
nch = args.nch
nker = args.nker

network = args.network
learning_type = args.learning_type

result_dir_train = os.path.join(result_dir, 'train')
result_dir_val = os.path.join(result_dir, 'val')
result_dir_test = os.path.join(result_dir, 'test')

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir_train, 'png'))
    os.makedirs(os.path.join(result_dir_val, 'png'))

    os.makedirs(os.path.join(result_dir_test, 'png'))
    os.makedirs(os.path.join(result_dir_test, 'numpy'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## 네트워크 학습
if mode == 'train':
    transform_train = transforms.Compose([RandomCrop(shape=(ny,nx)), Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
    transform_val = transforms.Compose([RandomCrop(shape=(ny,nx)), Normalization(mean=0.5, std=0.5), ToTensor()])

    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform_train, task=task, opts=opts)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform_val, task=task, opts=opts)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)
else:
    transform_test = transforms.Compose([RandomCrop(shape=(ny,nx)), Normalization(mean=0.5, std=0.5), ToTensor()])

    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform_test, task=task, opts=opts)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)

## 네트워크 생성
if network == 'unet':
    net = UNet(nch=nch, nker=nker, norm='bnorm', learning_type=learning_type).to(device)
# elif network == 'resnet':
#     net = ResNet().to(device)

## 손실함수 정의
fn_loss = nn.MSELoss().to(device)

## Optimizer 설정
optim = torch.optim.Adam(net.parameters(), lr=lr)

## 그밖에 부수적인 functions 설정
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
# fn_class = lambda x: 1.0 * (x > 0.5)

cmap = None

## Tensorboard를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

## 네트워크 학습시키기
st_epoch = 0
net, optim, st_epoch = load(ckpt_dir, net, optim)

if mode == 'train':
    if train_continue == 'on':
        net, optim, st_epoch = load(ckpt_dir, net, optim)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # backward pass
            optim.zero_grad()

            loss = fn_loss(output, label)
            loss.backward()

            optim.step()

            # 손실함수 계산
            loss_arr += [loss.item()]
            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                    (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))
            
            # Tensorboard 저장하기
            label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

            input = np.clip(input, a_min=0, a_max=1)
            output = np.clip(output, a_min=0, a_max=1)

            id = num_batch_train * (epoch - 1) + batch

            if epoch % 10 == 0:
                plt.imsave(os.path.join(result_dir_train, 'png', '%04d_label.png' % id), label[0], cmap=cmap)
                plt.imsave(os.path.join(result_dir_train, 'png', '%04d_input.png' % id), input[0], cmap=cmap)
                plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output.png' % id), output[0], cmap=cmap)

            # writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            # writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            # writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')


        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        # validation set
        with torch.no_grad():
            net.eval()
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                # 손실함수 계산
                loss = fn_loss(output, label)

                loss_arr += [loss.item()]

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                    (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))
                
                # Tensorboard 저장
                label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

                input = np.clip(input, a_min=0, a_max=1)
                output = np.clip(output, a_min=0, a_max=1)

                id = num_batch_val * (epoch - 1) + batch

                if epoch % 10 == 0:
                    plt.imsave(os.path.join(result_dir_val, 'png', '%04d_label.png' % id), label[0], cmap=cmap)
                    plt.imsave(os.path.join(result_dir_val, 'png', '%04d_input.png' % id), input[0], cmap=cmap)
                    plt.imsave(os.path.join(result_dir_val, 'png', '%04d_output.png' % id), output[0], cmap=cmap)

                # writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                # writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                # writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
            
            writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

            if epoch % 50 == 0:
                save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)


    writer_train.close()
    writer_val.close()
else:
    net, optim, st_epoch = load(ckpt_dir, net, optim)

    # validation set
    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # 손실함수 계산
            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                    (batch, num_batch_test, np.mean(loss_arr)))
            
            label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

            for j in range(label.shape[0]):
                id = num_batch_test * (batch - 1) + j

                label_ = label[j]
                input_ = input[j]
                output_ = output[j]

                np.save(os.path.join(result_dir_test, 'numpy', '%04d_label.npy' % id), label_)
                np.save(os.path.join(result_dir_test, 'numpy', '%04d_input.npy' % id), input_)
                np.save(os.path.join(result_dir_test, 'numpy', '%04d_output.npy' % id), output_)

                label_ = np.clip(label_, a_min=0, a_max=1)
                input_ = np.clip(input_, a_min=0, a_max=1)
                output_ = np.clip(output_, a_min=0, a_max=1)

                plt.imsave(os.path.join(result_dir_test, 'png', '%04d_label.png' % id), label_, cmap=cmap)
                plt.imsave(os.path.join(result_dir_test, 'png', '%04d_input.png' % id), input_, cmap=cmap)
                plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output.png' % id), output_, cmap=cmap)

    print("AVG TEST: BATCH %04d / %04d | LOSS %.4f" %
                    (batch, num_batch_test, np.mean(loss_arr)))
