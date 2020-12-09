# -*- coding: utf-8 -*-
# @Time    : 2020/12/9 下午1:05
# @Author  : Chaos
# @FileName: chaper_6_lanModel.py
# @Software: PyCharm
# @Email   : life0531@foxmail.com
'''语言模型数据集'''
import torch
import random
import zipfile

def load_data_jay_lyrics():
    '''读取数据集'''
    with zipfile.ZipFile('../datasets/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[:10000]


    # print(corpus_chars[:10000])

    '''建立字符索引/构造词典'''
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    # print(vocab_size)

    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    # sample = corpus_indices[:20]
    # print('chars: {}'.format("".join(idx_to_char[idx] for idx in sample)), end="\n")
    # print('indices: {}'.format(sample))
    return corpus_indices, char_to_idx, idx_to_char, vocab_size

'''随即采样'''
def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)
    
    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos+num_steps]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for i in range(epoch_size):
            # 每次读取batch_size个随即样本
            i = i * batch_size
            batch_indices = example_indices[i: i+batch_size]
            X = [_data(j*num_steps) for j in batch_indices]
            Y = [_data(j*num_steps+1) for j in batch_indices]
            yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)
