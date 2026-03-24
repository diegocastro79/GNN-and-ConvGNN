import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from pathlib import Path
from constants.constants import GnnModelArgs
from models.abstract_model import AbstractGNN


class ConvLayer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            feat_dropout: float,
            layer: int,
            drop_edges: bool = False
    ):
        super().__init__()
        torch.manual_seed(0)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.feat_dropout = feat_dropout
        self.layer = layer
        self.conv = GCNConv(in_dim, out_dim)
        self.drop_edges = drop_edges

    def forward(self, input_args: tuple):
        x, edge_index = input_args
        if self.training and self.layer == 0 and self.drop_edges:
            indices = random.sample(range(edge_index.shape[1]), int(0.95 * edge_index.shape[1])) # randomly discard 5% of connected nodes
            edge_index = edge_index[:, indices]
        emb = self.conv(x, edge_index)
        emb = F.relu(emb)
        emb = F.dropout(emb, p=self.feat_dropout, training=self.training)
        return emb, edge_index


class ConvGNN(AbstractGNN):
    def __init__(
            self,
            gnn_args: dict[GnnModelArgs, int | float | bool | list]
    ):
        super().__init__()
        torch.manual_seed(0)
        dim_list: list[int] = gnn_args[GnnModelArgs.DimList] # contains the number of features for each convolutional layer
        num_classes: int = gnn_args[GnnModelArgs.NumClasses]
        feat_dropout: float = gnn_args[GnnModelArgs.FeatDropout]
        drop_edges: bool = gnn_args[GnnModelArgs.DropEdges]
        num_layers = len(dim_list) - 1

        conv_layers = nn.ModuleList(
            [
                ConvLayer(
                    in_dim=dim_list[i],
                    out_dim=dim_list[i+1],
                    feat_dropout=feat_dropout,
                    layer=i,
                    drop_edges=drop_edges
                ) for i in range(num_layers)
            ]
        )
        self.conv_block = nn.Sequential(*conv_layers)
        self.last_layer = GCNConv(dim_list[-1], num_classes)

    def forward(self, x, edge_index):
        x, edge_index = self.conv_block((x, edge_index))
        return self.last_layer(x, edge_index)
