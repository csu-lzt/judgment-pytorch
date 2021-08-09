import torch
import torch.nn as nn
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MemoryNetworkLayer(nn.Module):
    def __init__(self):
        super(MemoryNetworkLayer, self).__init__()
        self.softmax = nn.Softmax()

    def forward(self, memory_slot):
        if memory_slot.size()[0] == 1:
            return torch.cat((memory_slot, memory_slot), 1)
        u = memory_slot[-1:]
        m = memory_slot[:-1]
        um = u @ m.t()
        p = self.softmax(um)
        o = p @ m
        return torch.cat((u, o), 1)


class MemoryNetwork:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.num_sentence = 0  # 计数句子序号
        self.num_judgment = 0  # 计数文书序号
        self.memory_slot = torch.empty(1, hidden_size).to(device)
        self.memory_network_layer = MemoryNetworkLayer().to(device)

    def get_memory_embedding(self, input_embedding, mode_length, mode):
        length_list = mode_length[mode]  # 取哪个数据集  的  文本句子数量
        # 开始存储记忆向量
        if self.memory_slot.size() == input_embedding.size():  # 判断记忆槽是否为空
            self.memory_slot = input_embedding
        else:
            self.memory_slot = torch.cat((self.memory_slot, input_embedding), 0)
        memory_embedding = self.memory_network_layer(self.memory_slot)
        # 开始计数
        self.num_sentence += 1
        if self.num_sentence >= length_list[self.num_judgment]:
            self.num_sentence = 0
            self.num_judgment += 1
            self.memory_slot = torch.empty(1, self.hidden_size).to(device)
        if self.num_judgment > len(length_list):
            self.memory_init()
        return memory_embedding

    def memory_init(self):
        self.num_sentence = 0  # 计数句子序号
        self.num_judgment = 0  # 计数文书序号
        self.memory_slot = torch.empty(1, self.hidden_size).to(device)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MemoryNetworkLayer().to(device)
    model(torch.rand(4, 768))
