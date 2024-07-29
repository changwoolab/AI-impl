import os
import numpy as np

import torch
import torch.nn as nn

## HyperParameters
lr = 1e-3
batch_size = 64
num_epoch = 10

ckpt_dir = './checkpoint'
log_dir = './log'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(256*6*6, 4096)
        self.relu6 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(4096, 1000)

    def forward(self, x):
        self.conv1_out = self.conv1(x)
        self.relu1_out = self.relu1(self.conv1_out)
        self.pool1_out = self.pool1(self.relu1_out)

        self.conv2_out = self.conv2(self.pool1_out)
        self.relu2_out = self.relu2(self.conv2_out)
        self.pool2_out = self.pool2(self.relu2_out)

        self.conv3_out = self.conv3(self.pool2_out)
        self.relu3_out = self.relu3(self.conv3_out)

        self.conv4_out = self.conv4(self.relu3_out)
        self.relu4_out = self.relu4(self.conv4_out)

        self.conv5_out = self.conv5(self.relu4_out)
        self.relu5_out = self.relu5(self.conv5_out)
        self.pool3_out = self.pool3(self.relu5_out)

        self.fc1_out = self.fc1(self.pool3_out.view(-1, 256*6*6))
        self.relu6_out = self.relu6(self.fc1_out)
        self.drop1_out = self.drop1(self.relu6_out)
        
        self.fc2_out = self.fc2(self.drop1_out)
        self.relu7_out = self.relu7(self.fc2_out)
        self.drop2_out = self.drop2(self.relu7_out)

        self.fc3_out = self.fc3(self.drop2_out)

        return self.fc3_out