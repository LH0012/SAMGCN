import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias if self.bias is not None else output


class decoder(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(nhid2, nhid1),
            nn.BatchNorm1d(nhid1),
            nn.ReLU()
        )
        self.pi = nn.Linear(nhid1, nfeat)
        self.disp = nn.Linear(nhid1, nfeat)
        self.mean = nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class SAMGCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout):
        super(SAMGCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.att = Attention(nhid2)
        self.MLP = nn.Sequential(nn.Linear(nhid2, nhid2))
        self.ZINB = decoder(nfeat, nhid1, nhid2)
        self.dropout = dropout
        #用于存储注意力权重
        self.attention_weights = None

    def forward(self, stg, oridata, augdata, nfadj, nsadj):
        emb1 = self.gc2(oridata, stg)
        emb2 = self.gc2(augdata, stg)
        emb3 = self.gc2(oridata, nfadj)
        emb4 = self.gc2(oridata, nsadj)
        # emb1 = self.gc2(oridata, stg)
        # #emb2 = self.gc2(augdata, stg)
        # emb3 = self.gc2(oridata, nfadj)
        # emb4 = self.gc2(oridata, nsadj)

        emb = torch.stack([emb1, (emb3 + emb4), emb2], dim=1)
        #emb = torch.mean(torch.stack([emb1, (emb3 + emb4), emb2], dim=1), dim=1)#取消注意力机制
        emb, att = self.att(emb)
        # 存储注意力权重
        self.attention_weights = att.detach().cpu().numpy()  # 转换为 NumPy 数组
        emb = self.MLP(emb)

        pi, disp, mean = self.ZINB(emb)
        return emb1, emb2, emb, pi, disp, mean
