# -*- coding: utf-8 -*-
# @Time    : 2020/12/7 下午1:35
# @Author  : Chaos
# @FileName: chapter_3_dropout.py
# @Software: PyCharm
# @Email   : life0531@foxmail.com
'''dropout的两种实现'''
import torch
import numpy as np

def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃
    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) < keep_prob).float()

    return mask * X / keep_prob

'''从零实现'''
# num_inputs, num_outputs, num_hidden1, num_hidden2 = 784, 10, 256, 256
# W1 = torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hidden1)), dtype=torch.float, requires_grad=True)
# b1 = torch.zeros(num_hidden1, requires_grad=True)
# W2 = torch.tensor(np.random.normal(0, 0.01, size=(num_hidden1, num_hidden2)), dtype=torch.float, requires_grad=True)
# b2 = torch.zeros(num_hidden2, requires_grad=True)
# W3 = torch.tensor(np.random.normal(0, 0.01, size=(num_hidden2, num_outputs)), dtype=torch.float, requires_grad=True)
# b3 = torch.zeros(num_outputs, requires_grad=True)
#
# params = [W1, b1, W2, b2, W3, b3]12
#
# drop_p1, drop_p2 = 0.2, 0.5
# def net(X, is_training=True):
#     X = X.view(-1, num_inputs)
#     H1 = (torch.matmul(X, W1) + b1).relu()
#     if is_training:
#         H1 = dropout(H1, drop_p1)
#     H2 = (torch.matmul(H1, W2) + b2).relu()
#     if is_training:
#         H2 = dropout(H2, drop_p2)
#
#     return torch.matmul(H2, W3) + b3

'''简洁实现'''
