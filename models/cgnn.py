import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from pathlib import Path


class ConvLayer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            drop: float,
            layer: int,
            drop_edges: bool = False
    ):
        super().__init__()
        torch.manual_seed(0)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.drop = drop
        self.layer = layer
        self.conv = GCNConv(in_dim, out_dim)
        self.drop_edges = drop_edges

    def forward(self, input_args: tuple):
        x, edge_index = input_args
        if self.training and self.layer == 0 and self.drop_edges:
            indices = random.sample(range(edge_index.shape[1]), int(0.95 * edge_index.shape[1]))
            edge_index = edge_index[:, indices]
        emb = self.conv(x, edge_index)
        emb = F.relu(emb)
        emb = F.dropout(emb, p=self.drop, training=self.training)
        return emb, edge_index


class ConvGNN(nn.Module):
    def __init__(
            self,
            dim_list: list[int], # contains the number of features for all layers except for the last one
            num_classes: int,
            drop: float = 0.5,
            drop_edges: bool = False
    ):
        super().__init__()
        torch.manual_seed(0)
        self.dim_list = dim_list
        self.num_classes = num_classes
        self.last_conv = GCNConv(dim_list[-1], num_classes)
        num_layers = len(dim_list) - 1
        conv_layers = nn.ModuleList(
            [
                ConvLayer(dim_list[i], dim_list[i+1], drop, i, drop_edges=drop_edges)
                for i in range(num_layers)
            ]
        )
        self.conv_block = nn.Sequential(*conv_layers)

    def forward(self, x, edge_index):
        emb, edge_index = self.conv_block((x, edge_index))
        return self.last_conv(emb, edge_index)


def save_upload_model_state(
        model: ConvGNN,
        path: Path,
        trial: int = None,
        upload: bool = False,
) -> ConvGNN | None:
    if trial is not None:
        file_path = path / f"best_model_trial_{trial}.pt"
    else:
        file_path = path / f"best_model.pt"
    if upload:
        if Path(file_path).exists():
            model.load_state_dict(torch.load(file_path, weights_only=True))
            return model
        raise ValueError(f"The file {path} does not exists!\n")
    torch.save(model.state_dict(), file_path)