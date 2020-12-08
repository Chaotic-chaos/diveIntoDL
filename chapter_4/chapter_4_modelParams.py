# -*- coding: utf-8 -*-
# @Time    : 2020/12/8 上午9:54
# @Author  : Chaos
# @FileName: chapter_4_modelParams.py
# @Software: PyCharm
# @Email   : life0531@foxmail.com
'''模型参数的访问、初始化和共享'''
import torch
from torch import nn
from torch.nn import init

net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))

# print(net)
# X = torch.rand(2, 4)
# Y = net(X).sum

'''访问模型参数'''
# print(type(net.named_parameters()))
# for name, param in net.named_parameters():
#     print("{} : {}".format(name, param.size()))

'''模型参数初始化'''
# for name, param in net.named_parameters():
#     if 'weight' in name:
#         init.normal_(param, mean=0, std=0.01)
#         print(name, param.data)
#
#     if 'bias' in name:
#         init.constant_(param, val=0)
#         print(name, param.data)

'''自定义初始化方法'''
def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10, 10)
        tensor *= (tensor.abs() >= 5).float()

for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print('{} : {}'.format(name, param.data))