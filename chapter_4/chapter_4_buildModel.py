# -*- coding: utf-8 -*-
# @Time    : 2020/12/7 下午4:09
# @Author  : Chaos
# @FileName: chapter_4_buildModel.py
# @Software: PyCharm
# @Email   : life0531@foxmail.com
'''深度学习计算'''
import torch
from torch import nn

'''模型构造'''
class MLP(nn.Module):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256) # 隐藏层
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10) # 输出层

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

# X = torch.rand(2, 784)
# net = MLP()
# print(net)
# print(net(X))

'''构造复杂的模型'''
class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)

        self.rand_weight = torch.rand((20,20), requires_grad=False) # 不可训练参数（常参）
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)

        # 复用全连接层，等价于两个全连接层共享参数
        x = self.linear(x)
        # 控制流，需要调用item函数来返回标量进行比较
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()


# x = torch.rand(2, 20)
# net = FancyMLP()
# print(net)
# print(net(x))

'''嵌套调用sequential和FancyMLP'''
class NestMLP(nn.Module):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)

        self.net = nn.Sequential(nn.Linear(40, 30), nn.ReLU())

    def forward(self, x):
        return self.net(x)

net = nn.Sequential(NestMLP(), nn.Linear(30, 20), FancyMLP())

x = torch.rand(2, 40)
print(net)
print(net(x))