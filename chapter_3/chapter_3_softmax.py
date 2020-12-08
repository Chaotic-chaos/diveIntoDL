# -*- coding: utf-8 -*-
# @Time    : 2020/12/6 下午2:04
# @Author  : Chaos
# @FileName: chapter_3_softmax.py
# @Software: PyCharm
# @Email   : life0531@foxmail.com
'''softmax回归'''
import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
from chapter_3_datasets import load_data_fashion_mnist


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        if isinstance(net, torch.nn.Module):
            net.eval()  # 评估模式, 这会关闭dropout
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            net.train()  # 改回训练模式
        else:  # 自定义的模型
            if ('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                # 将is_training设置成False
                acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
            else:
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


def train_ch3(net, train_iter, test_iter, loss, epochs, batc_size, params=None, lr=None, optimizer=None):
    for epoch in range(epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                raise
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch: {}, loss: {:.4f}, train acc: {:.3f}, test acc: {:.3f}'.format(epoch+1, train_l_sum/n, train_acc_sum/n, test_acc))

if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

    num_inputs = 784
    num_outputs = 10

    class LinearNet(nn.Module):
        def __init__(self, num_inputs, num_outputs):
            super(LinearNet, self).__init__()
            self.linear = nn.Linear(num_inputs, num_outputs)

        def forward(self, x):
            y = self.linear(x.view(x.shape[0], -1))
            return y

    net = LinearNet(num_inputs, num_outputs)

    init.normal_(net.linear.weight, mean=0, std=0.01)
    init.constant_(net.linear.bias, val=0)

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    
    epochs = 5
    train_ch3(net, train_iter, test_iter, loss, epochs, batch_size, None, None, optimizer=optimizer)