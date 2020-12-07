# -*- coding: utf-8 -*-
# @Time    : 2020/12/7 上午10:08
# @Author  : Chaos
# @FileName: chapter_3_mutisensor.py
# @Software: PyCharm
# @Email   : life0531@foxmail.com
import torch
from torch import nn
from torch.nn import init
import numpy as np
from chapter_3_datasets import load_data_fashion_mnist
import chapter_3_softmax

'''对x的形状转换功能'''
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x.shape: （barch, *, *, ...）
        return x.view(x.shape[0], -1)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

net = nn.Sequential(
    FlattenLayer(),
    nn.Linear(num_inputs, num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens, num_outputs),
)

for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

epochs = 5
chapter_3_softmax.train_ch3(net, train_iter, test_iter, loss, epochs, batch_size, None, None, optimizer)
