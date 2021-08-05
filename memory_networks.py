import torch
import torch.nn as nn
from torchsummary import summary


class MemoryNetwork(nn.Module):
    def __init__(self):
        super(MemoryNetwork, self).__init__()
        self.softmax = nn.Softmax()

    def forward(self, memory_slot):
        if memory_slot.size()[0] == 1:
            return memory_slot
        u = memory_slot[-1:]
        m = memory_slot[:-1]
        um = u @ m.t()
        p = self.softmax(um)
        o = p @ m
        return torch.cat((u, o), 1)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MemoryNetwork().to(device)
    model(torch.rand(4, 768))
