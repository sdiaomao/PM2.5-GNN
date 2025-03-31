import torch
from torch import nn
from model.cells import GRUCell
from torch.nn import Sequential, Linear, Sigmoid
import numpy as np
from torch_scatter import scatter_add#, scatter_sub  # no scatter sub in lastest PyG
from torch.nn import functional as F
from torch.nn import Parameter


class GraphGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=8, dropout=0.6):
        """
        参数:
            in_dim (int): 节点特征输入维度
            hidden_dim (int): 第一层 GAT 隐藏维度
            out_dim (int): 输出节点特征维度
            heads (int): 第一层多头注意力的头数
            dropout (float): dropout 比例
        """
        super(GraphGAT, self).__init__()
        # 第一层 GAT 使用多头注意力，并将多个头的输出拼接
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        # 第二层 GAT 将第一层的所有头的输出作为输入（hidden_dim * heads），
        # 输出时采用单头（concat=False）实现输出维度 out_dim
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        """
        前向传播:
            x: 节点特征张量，形状 [num_nodes, in_dim] 或批处理后所有节点的拼接
            edge_index: 边索引张量，形状 [2, num_edges]
        返回:
            输出节点特征，形状 [num_nodes, out_dim]
        """
        # 第一层：注意力计算及拼接所有头的输出，然后使用激活函数（这里采用 ELU）
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        # 第二层：输出单头结果（直接输出 out_dim 维的节点特征）
        x = self.gat2(x, edge_index)
        return x


class PM25_GNN(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device, edge_index, edge_attr, wind_mean, wind_std):
        super(PM25_GNN, self).__init__()

        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size

        self.in_dim = in_dim
        self.hid_dim = 64
        self.out_dim = 1
        self.gnn_out = 13

        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)
        self.graph_gnn = GraphGAT(in_dim, hidden_dim, out_dim, heads=8, dropout=0.6)
        self.gru_cell = GRUCell(self.in_dim + self.gnn_out, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, pm25_hist, feature):
        pm25_pred = []
        h0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0
        xn = pm25_hist[:, -1]
        for i in range(self.pred_len):
            x = torch.cat((xn, feature[:, self.hist_len + i]), dim=-1)

            xn_gnn = x
            xn_gnn = xn_gnn.contiguous()
            xn_gnn = self.graph_gnn(xn_gnn)
            x = torch.cat([xn_gnn, x], dim=-1)

            hn = self.gru_cell(x, hn)
            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            xn = self.fc_out(xn)
            pm25_pred.append(xn)

        pm25_pred = torch.stack(pm25_pred, dim=1)

        return pm25_pred
