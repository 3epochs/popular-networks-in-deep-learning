import sys
import numpy
import torch
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader


def get_network(args, use_gpu=True):

    if args.net == 'vgg11':
        from VGG.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg13':
        from VGG.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg16':
        from VGG.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg19':
        from VGG.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'googlenet':
        from GoogleNet.googlenet import googlenet
        net = googlenet()
    elif args.net == 'alexnet':
        from AlexNet.alexnet import alexnet
        net = alexnet()
    elif args.net == 'resnet18':
        from ResNet.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from ResNet.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from ResNet.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from ResNet.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from ResNet.resnet import resnet152
        net = resnet152()
    else:
        print('The network you entered is not supported yet. \n PR if you have implemented it.')
        sys.exit()
    if use_gpu:
        net = net.cuda()
    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    cifar100_training = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=False, transform=transform_train
    )
    cifar100_training_dataloader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size
    )
    return cifar100_training_dataloader


def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    cifar100_test_dataloader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size
    )
    return cifar100_test_dataloader


def compute_mean_std(cifar100_dataset):
    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)
    return mean, std


class WarmUpLR(_LRScheduler):

    def __init__(self, optimizer, total_iters, last_epoch=1):

        self.total_iters = total_iters
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
