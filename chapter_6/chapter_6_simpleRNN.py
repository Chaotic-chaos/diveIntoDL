# -*- coding: utf-8 -*-
# @Time    : 2020/12/12 下午3:18
# @Author  : Chaos
# @FileName: chapter_6_simpleRNN.py
# @Software: PyCharm
# @Email   : life0531@foxmail.com
'''RNN的简洁实现'''

import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

from chapter_6.chaper_6_lanModel import load_data_jay_lyrics, data_iter_consecutive

device = ('cuda' if torch.cuda.is_available() else 'cpu')

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()

'''定义模型'''
num_hiddens = 256
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)

# num_steps = 35
# batch_size = 2
# state = None
# X = torch.rand(num_steps, batch_size, vocab_size)
# Y, state_new = rnn_layer(X, state)
# print(Y.shape, len(state_new), state_new[0].shape)

def one_hot(x, n_class, dtype=torch.float32):
    # X shape: (batch), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res

def to_onehot(X, n_class):
    # X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]

class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state):
        # 获取one-hot向量表示
        X = to_onehot(inputs, self.vocab_size)
        Y, self.state = self.rnn(torch.stack(X), state)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state

'''训练模型'''
# 定义预测函数
def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char, char_to_idx):
    state = None
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars+len(prefix)-1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        if state is not None:
            if isinstance(state, tuple):
                state = (state[0].to(device), state[1].to(device))
            else:
                state.to(device)

        (Y, state) = model(X, state)
        if t < len(prefix)-1:
            output.append(char_to_idx[prefix[t+1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])

model = RNNModel(rnn_layer, vocab_size).to(device)
# print(predict_rnn_pytorch('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx))

def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device)
    for param in params:
        norm += (param.grad.data ** 2).sum
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)

# def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device, corpus_indices, idx_to_char, char_to_idx, num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes):
#     loss = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     model.to(device)
#     state = None
#     for epoch in range(num_epochs):
#         l_sum, n, start = 0.0, 0, time.time()
#         data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps, device)
#         for X, Y in data_iter:
#             if state is not None:
#                 if isinstance(state, tuple):
#                     state = (state[0].detach(), state[1].detach())
#                 else:
#                     state = state.detach()
#             (output, state) = model(X, state)
#             y = torch.transpose(Y, 0, 1).contiguous().view(-1)
#             l = loss(output, y.long())
#
#             optimizer.zero_grad()
#             l.backward()
#             # 梯度裁剪
#             grad_clipping(model.parameters(), clipping_theta, device)
#             optimizer.step()
#             l_sum += l.item() * y.shape[0]
#             n += y.shape[0]
#
#         try:
#             perplexity = math.exp(l_sum / n)
#         except OverflowError:
#             perplexity = float('inf')
#         if (epoch + 1) % pred_period == 0:
#             print('epoch {}, perplexity {}, time, {:.2f} sec'.format(epoch+1, perplexity, time.time()-start))
#             for prefix in prefixes:
#                 print(' -', predict_rnn_pytorch(
#                     prefix, pred_len, model, vocab_size, device, idx_to_char,
#                     char_to_idx))

def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 10, time.time()
        data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps, device) # 相邻采样
        for X, Y in data_iter:
            if state is not None:
                # 使用detach函数从计算图分离隐藏状态, 这是为了
                # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                if isinstance (state, tuple): # LSTM, state:(h, c)
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()

            (output, state) = model(X, state) # output: 形状为(num_steps * batch_size, vocab_size)

            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # batch * num_steps 的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(output, y.long())

            optimizer.zero_grad()
            l.backward()
            # 梯度裁剪
            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            perplexity = float('inf')
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, perplexity, time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_pytorch(
                    prefix, pred_len, model, vocab_size, device, idx_to_char,
                    char_to_idx))

num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2 # 注意这里的学习率设置
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                            corpus_indices, idx_to_char, char_to_idx,
                            num_epochs, 6, lr, clipping_theta,
                            batch_size, pred_period, pred_len, prefixes)