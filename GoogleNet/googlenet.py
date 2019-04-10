import torch
import torch.nn as nn


class Inception(nn.Module):

    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reudce, n5x5, pool_proj):
        super(Inception, self).__init__()

        # 1x1 conv branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(input_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )

        # 1x1 conv -> 3x3 conv branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        # 1x1 conv -> 5x5 conv branch
        # can use two 3x3 stacking instead of 5x5:
        # same receptive field and fewer parameters
        self.branch3 = nn.Sequential(
            nn.Conv2d(input_channels, n5x5_reudce, kernel_size=1),
            nn.BatchNorm2d(n5x5_reudce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reudce, n5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )

        # 3x3 pooling -> 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(input_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch(4)], dim=1)


class GoogleNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(GoogleNet, self).__init__()
        self.prelayer = nn.Sequential(
            # 256 x 256 x 3
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            # 112 x 112 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 56 x 56 x 64
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            # 56 x 56 x 192
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # 28 x 28 x 192
        )

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        # 28 x 28 x 256
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        # 28 x 28 x 480

        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        # 14 x 14 x 480

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        # 14 x 14 x 512
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        # 14 x 14 x 512
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        # 14 x 14 x 512
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        # 14 x 14 x 528
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
        # 14 x 14 x 832

        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        # 7 x 7 x 832
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        # 7 x 7 x 832
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
        # 7 x 7 x 1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 1 x 1 x 1024
        self.droupout = nn.Dropout2d(0.4)
        # 1 x 1 x 1024
        self.fc = nn.Linear(1024, num_classes)
        # 1 x 1 x 1000

    def forward(self, x):
        output = self.prelayer(x)

        output = self.a3(output)
        output = self.b3(output)

        output = self.maxpool1(output)

        output = self.a4(output)
        output = self.b4(output)
        output = self.c4(output)
        output = self.d4(output)
        output = self.e4(output)

        output = self.maxpool2(output)

        output = self.a5(output)
        output = self.b5(output)

        output = self.avgpool(output)

        output = self.droupout(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def googlenet():
    return GoogleNet()
