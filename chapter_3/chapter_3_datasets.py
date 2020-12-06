# -*- coding: utf-8 -*-
# @Time    : 2020/12/6 下午2:51
# @Author  : Chaos
# @FileName: chapter_3_datasets.py
# @Software: PyCharm
# @Email   : life0531@foxmail.com
'''Fashion-MNIST以及torchvision的使用'''
import sys

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

def load_data_fashion_mnist(batch_size):
    '''调用数据集'''
    mnist_train = torchvision.datasets.FashionMNIST(root='../datasets', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='../datasets', train=False, download=True, transform=transforms.ToTensor())
    # print(type(mnist_train))
    # print("训练集长度： {}， 测试集长度： {}".format(len(mnist_train), len(mnist_test)))
    # feature, label = mnist_train[0]
    # print(feature.shape, label)
    # batch_size = 256
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_iter, test_iter

if __name__ == '__main__':
    start_time = time.time()
    train_iter, test_iter = load_data_fashion_mnist()
    for X, y in train_iter:
        continue
    print("{:.2f} seconds".format(time.time() - start_time))