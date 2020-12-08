# -*- coding: utf-8 -*-
# @Time    : 2020/12/8 下午1:27
# @Author  : Chaos
# @FileName: chapter_4_customerLayer.py
# @Software: PyCharm
# @Email   : life0531@foxmail.com
'''自定义神经网络中的层'''
import torch
from torch import nn

'''不含参数的自定义层'''
class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__()
    def forward(self, x):
        return x - x.mean()

# layer = CenteredLayer()
# print(layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)))


'''含模型参数的自定义层'''
class MyDense(nn.Module):
    def __init__(self):
        super(MyDense, self).__init__()

        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x

# net = MyDense()
# print(net)

