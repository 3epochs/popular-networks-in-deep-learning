import argparse

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.autograd import Variable

from _conf import settings
from utils import get_network, get_test_dataloader


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    args = parser.parse_args()

    net = get_network(args)

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    net.load_state_dict(torch.load(args.weights), args.gpu)
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    for n_iter, (images, labels) in enumerate(cifar100_test_loader):
        print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        output = net(images)
        _, pred = output.topk(5, 1, largest=True, sorted=True)

        labels = labels.view(labels.size(0), -1).expand_as(pred)
        correct = pred.eq(labels).float()

        correct_5 += correct[:, :5].sum()
        correct_1 += correct[:, :1].sum()
    print()
    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
