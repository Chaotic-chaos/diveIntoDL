# -*- coding: utf-8 -*-
# @Time    : 2020/10/13 下午9:18
# @Author  : Chaos
# @FileName: chapter_2.py
# @Software: PyCharm
# @Email   : life0531@foxmail.com
# 导入torch
import torch

# 创建一个5x3的未初始化Tensor
x = torch.empty(5, 3)
# print(x)

# 创建一个5x3的随即初始化的Tensor
y = torch.rand(5, 3)
# print(y)

# 创建一个5x3的long型全0Tensor
z = torch.zeros(5, 3, dtype=torch.long)
# print(z)

# 根据现有数据创建
l = torch.tensor([5.5, 3])
# print(l)

# 根据现有Tensor创建
'''
    1. 默认重用输入的Tensor的一些属性，如数据类型等
    2. 创建时自定义数据类型优先级更高
'''
m = x.new_ones(5, 3, dtype=torch.float64) # 返回的Tensor默认具有相同的torch.dtype及torch.device[注①]
# print(m)
n = torch.randn_like(x, dtype=torch.float) # 自定义新的数据类型
# print(n)

# 通过shape或者size()获取Tensor形状
'''
    1. 返回的torch.Size()是一个tuple
    2. 其支持tuple所有原生操作
'''
# print(x.size())xh
# print(x.shape)

'''item()函数的使用'''
x = torch.randn(1)
# print(x)
# print(x.item())

'''CPU和GPU之间的tensor传递'''
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)