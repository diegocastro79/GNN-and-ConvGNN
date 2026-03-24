from abc import ABC
import torch.nn as nn

class AbstractGNN(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index):
        raise NotImplementedError