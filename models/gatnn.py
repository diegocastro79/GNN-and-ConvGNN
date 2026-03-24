import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv
from models.abstract_model import AbstractGNN
from constants.constants import GnnModelArgs


class GatLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int, att_dropout: float = 0.05, feat_dropout: float = 0.5):
        super().__init__()
        self.feat_dropout = feat_dropout
        self.gatconv = GATConv(in_dim, out_dim, heads=heads, dropout=att_dropout, concat=True)

    def forward(self, input_args: tuple):
        x, edge_index = input_args
        x = self.gatconv(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.feat_dropout, training=self.training)
        return x, edge_index

class GAttNN(AbstractGNN):
    def __init__(
            self,
            gnn_args: dict[GnnModelArgs, int | float | bool | list]
    ):
        super().__init__()
        dim_list = gnn_args[GnnModelArgs.DimList] # contains the number of in-features for each attention layer
        heads_list = gnn_args[GnnModelArgs.AttHeadsList] # contains the number of heads for each attention layer
        if len(dim_list) != len(heads_list) + 1:
            raise RuntimeError("Dimensions mismatch. 'dim_list' should contain exactly one more element than 'heads_list'."
                             f"Got instead {len(dim_list)} for 'dim_list' and {len(heads_list)} for 'heads_list'.'")

        att_layers = nn.ModuleList(
            [
                GatLayer(
                    in_dim=dim_list[i] if i==0 else dim_list[i]*heads_list[i-1],
                    out_dim=dim_list[i+1],
                    heads=heads,
                    att_dropout=gnn_args[GnnModelArgs.AttDropout],
                    feat_dropout=gnn_args[GnnModelArgs.FeatDropout],
                ) for i, heads in enumerate(heads_list)
            ]
        )
        self.att_block = nn.Sequential(*att_layers)
        self.last_layer = nn.Linear(heads_list[-1]*dim_list[-1], gnn_args[GnnModelArgs.NumClasses])

    def forward(self, x, edge_index):
        x, _ = self.att_block((x, edge_index))
        return self.last_layer(x)



