import torch
import torch.nn as nn
import torch.nn.functional as F

""" @author: 3epochs
    base architecture:
    1. input image: samples_num, 1 (gray scale), 32, 32;
    2. convolution layer1: in_channel: 1, out_channel: 6, kernel_size: 5 * 5, followed by Relu non-linearity;
    3. subsample: max pooling, kernel_size: 2 * 2 ;
    4. convolution layer2: in_channel: 6, out_channel: 16, kernel_size: 5 * 5, followed by Relu non-linearity;
    5. subsample: max polling, kernel size: 2 * 2;
    6. fully connected layer: in_channel: 16 * 5 * 5, out_channel: 120, Relu;
    7. fully connected layer: in_channel: 120, out_channel: 84, Relu;
    8. fully connected layer: in_channel: 84, out_channel: 10."""


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(in_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x





