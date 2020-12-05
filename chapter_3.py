# 线性回归 - 从零实现
import torch
import numpy as np
import torch.utils.data as Data

# num_inputs = 2 # 特征数
# num_examples = 1000 # 样本数
# true_w = [2, -3.4] # 真实权重
# true_b = 4.2 # 真实偏置
# features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
# labels = true_w[0] * features[:, 0] + features[:, 1] + true_b
# labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
# 
# # 生成第二个特征与标签之间的散点图
# set_figsize()
# plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
from torch import nn

'''
    线性回归 - 简洁实现
        1. 利用torch提供的接口函数
'''

'''生成数据集'''
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0,  1, (num_examples, num_inputs)), dtype=torch.float) # 训练数据特征
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float) # 标签

'''读取数据'''
batch_size = 10
# 组合训练数据的特征和标签
dataset = Data.TensorDataset(features, labels)
# 随机读取batch_size
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
# 输出一个batch_size查看
# for X, y in data_iter:
#     print(X, y)
#     break

'''定义模型'''
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        '''
        模型初始化
        :param n_feature: 特征数量/维度
        '''
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1) # param1: 特征维度  param2: 输出维度

    # 前向传播
    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
# 打印网络结构
print(net)