# -*- coding: utf-8 -*-
# @Time    : 2020/12/8 下午2:22
# @Author  : Chaos
# @FileName: chapter_4_io.py
# @Software: PyCharm
# @Email   : life0531@foxmail.com
'''模型的存储和读取'''
import torch
from torch import nn

'''读写Tensor'''
x = torch.ones(3)
# print(x)
# torch.save(x, './x.pt')

# x2 = torch.load("./x.pt")
# print(x2)

y = torch.ones(4)
# torch.save([x, y], './xy.pt')
# xy_list = torch.load("./xy.pt")
# print(xy_list)

# torch.save({'x': x, 'y': y}, 'xy_dict.pt')
# xy = torch.load('./xy_dict.pt')
# print(xy)

'''读写模型'''
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)
        
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)
    
net = MLP()
# print(net.state_dict())

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# print(optimizer.state_dict())

'''
保存/加载模型
    1. 保存和加载模型参数state_dict(推荐)
    2. 保存和加载整个模型
'''
'''保存和加载模型参数'''
# torch.save(model.state_dict(), PATH) # 推荐的文件后缀是pt或pth
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))

