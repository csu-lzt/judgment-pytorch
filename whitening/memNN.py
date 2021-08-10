#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
第三步，对向量进行记忆网络模块操作
修改embedding_path和save_path
'''
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import load_data
from memory_networks import MemoryNetwork
from tqdm import tqdm


class EmbeddingsDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, embedding_path, data_path):
        self.data = np.load(embedding_path)
        _, self.labels, self.length = load_data(data_path, return_length=True)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_length_of_single_judgment(self):
        return self.length


class MemoryEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super(MemoryEmbedding, self).__init__()
        self.memory_network = MemoryNetwork(hidden_size)

    def forward(self, embeddings):
        memory_embeddings = self.memory_network.get_memory_embedding(embeddings, mode_length, 'train')
        return memory_embeddings


train_dataset = EmbeddingsDataset(embedding_path='whitening/embedding_avg_whiten.npy',
                                  data_path='data/classify_data/train_data.json')
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0,
                          pin_memory=True)
train_length = train_dataset.get_length_of_single_judgment()
mode_length = {'train': train_length}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MemoryEmbedding(hidden_size=768).to(device)

memory_embeddings = []
for batch in tqdm(train_loader):
    input = batch.to(device)
    embedding = model(input).cpu().numpy()
    memory_embeddings.extend(embedding)

memory_embeddings = np.array(memory_embeddings)
print(memory_embeddings.shape)
save_path = 'whitening/embedding_avg_whiten_memNN.npy'
np.save(save_path, memory_embeddings)
