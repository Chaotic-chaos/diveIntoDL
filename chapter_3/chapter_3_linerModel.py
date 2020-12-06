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
from torch import nn, optim
from torch.nn import init

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

'''定义模型：方法一'''
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
# print(net)


'''初始化模型参数(w, b)'''
init.normal_(net.linear.weight, mean=0, std=0.01) # param1: 要初始化的参数； param2: 均值； param3: 标准差
init.constant_(net.linear.bias, val=0) # param1: 要初始化的参数； param2: 初始化值

'''定义损失函数'''
loss = nn.MSELoss() # 均方误差损失

'''定义优化算法'''
optimizer = optim.SGD(net.linear.parameters(), lr=0.03) # param1: 优化参数; param2: 学习率
# print(optimizer)

'''训练模型'''
num_epochs = 1000
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零
        l.backward() # 反向传播
        optimizer.step() # 模型迭代
    print('epoch {}, loss: {}'.format(epoch, l.item()))
print("\n\n\n\n\n")
'''对比学到的模型与真实的模型参数'''
print("true_w: {}\nlearn_w: {}".format(true_w, net.linear.weight.data))
print('--------------------------------------------')
print("true_b: {}\nlearn_b: {}".format(true_b, net.linear.bias.data))