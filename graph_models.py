import torch.nn as nn
from torch_geometric.nn import GCNConv, GCN2Conv, GATConv, SAGEConv, GINConv, JumpingKnowledge
import torch.nn.functional as F
from torch_geometric.nn.norm import BatchNorm, LayerNorm, PairNorm
from torch_geometric.utils import dropout_edge
import torch.nn.functional as F

# Base Functions
def make_act(name: str):
    name = name.lower()
    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}.get(name, nn.ReLU)()

def make_norm(name: str, dim: int):
    name = name.lower()
    if name == "batch": return BatchNorm(dim)
    if name == "layer": return LayerNorm(dim)
    if name == "pair":  return PairNorm()
    return nn.Identity()

# Models
class GCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        dropout: float = 0.5,
        edge_dropout_p: float = 0.1,
        activation: str = "relu",
        norm: str = "batch",
        residual: bool = True,
        jk: str = "last",
    ):
        super().__init__()
        assert num_layers >= 2, "num_layers must be >= 2"

        self.dropout = dropout
        self.edge_dropout_p = edge_dropout_p
        self.act = make_act(activation)
        self.norm_name = norm.lower()
        self.residual = residual
        self.jk_mode = jk

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.norms.append(make_norm(norm, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(make_norm(norm, hidden_channels))

        self.jk = None if jk == "last" else JumpingKnowledge(mode=jk)

        head_in = hidden_channels if jk != "cat" else hidden_channels * (num_layers - 1)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_channels),
            make_act(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, edge_index):
        if self.training and self.edge_dropout_p > 0:
            ei, _ = dropout_edge(edge_index, p=self.edge_dropout_p, force_undirected=False, training=True)
        else:
            ei = edge_index

        hs = []
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_prev = x
            x = conv(x, ei)
            x = norm(x) if self.norm_name != "pair" else norm(x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.residual and i > 0:
                x = x + h_prev
            hs.append(x)

        x = hs[-1] if self.jk is None else self.jk(hs[1:])
        return self.head(x)


class GCNII(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 16,
        dropout: float = 0.5,
        alpha: float = 0.1,
        theta: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.layers = nn.ModuleList()

        self.in_lin = nn.Linear(in_channels, hidden_channels)

        for _ in range(num_layers):
            self.layers.append(GCN2Conv(channels=hidden_channels, alpha=alpha, theta=theta, layer=_ + 1))

        self.out_lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x0 = F.dropout(x, p=self.dropout, training=self.training)
        x0 = F.relu(self.in_lin(x0))
        x  = x0
        for conv in self.layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(conv(x, x0, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out_lin(x)


class GAT(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=4,
        heads=4,
        dropout=0.5,
        edge_dropout_p=0.1,
        activation="elu",
        norm="batch",
        residual=True,
        jk="last"
    ):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout
        self.edge_dropout_p = edge_dropout_p
        self.residual = residual
        self.jk_mode = jk
        self.act = make_act(activation)

        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))
        self.norms.append(make_norm(norm, hidden_channels * heads))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))
            self.norms.append(make_norm(norm, hidden_channels * heads))

        self.jk = None if jk == "last" else JumpingKnowledge(mode=jk)

        head_in = hidden_channels * heads if jk != "cat" else (hidden_channels * heads) * (num_layers - 1)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_channels),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index):
        if self.training and self.edge_dropout_p > 0:
            edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout_p, training=True)

        hs = []
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_prev = x
            x = conv(x, edge_index)
            x = norm(x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.residual and i > 0:
                x = x + h_prev
            hs.append(x)

        x = hs[-1] if self.jk is None else self.jk(hs[1:])
        return self.head(x)


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=3,
        dropout=0.5,
        activation="relu",
        norm="batch",
        residual=True,
        jk="last"
    ):
        super().__init__()
        self.dropout = dropout
        self.residual = residual
        self.act = make_act(activation)

        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(make_norm(norm, hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(make_norm(norm, hidden_channels))

        self.jk = None if jk == "last" else JumpingKnowledge(mode=jk)
        self.jk_mode = jk

        if jk == "cat":
            head_in = hidden_channels * num_layers
        else:
            head_in = hidden_channels

        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_channels),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index):
        hs = []
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_prev = x
            x = conv(x, edge_index)
            x = norm(x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.residual and i > 0:
                x = x + h_prev
            hs.append(x)

        if self.jk is None:
            x = hs[-1]
        else:
            x = self.jk(hs)

        return self.head(x)


class GIN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=4,
        dropout=0.5,
        activation="relu",
        norm="batch",
        residual=True,
        jk="last"
    ):
        super().__init__()
        self.dropout = dropout
        self.residual = residual
        self.act = make_act(activation)

        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        nn_linear = lambda: nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            self.act,
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(GINConv(nn.Sequential(nn.Linear(in_channels, hidden_channels), self.act, nn.Linear(hidden_channels, hidden_channels))))
        self.norms.append(make_norm(norm, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(nn_linear()))
            self.norms.append(make_norm(norm, hidden_channels))

        self.jk = None if jk == "last" else JumpingKnowledge(mode=jk)

        head_in = hidden_channels if jk != "cat" else hidden_channels * (num_layers - 1)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_channels),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index):
        hs = []
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_prev = x
            x = conv(x, edge_index)
            x = norm(x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.residual and i > 0:
                x = x + h_prev
            hs.append(x)

        x = hs[-1] if self.jk is None else self.jk(hs[1:])
        return self.head(x)