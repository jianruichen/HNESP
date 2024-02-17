import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, noutput, dropout1, node_num, n_feature, n_hidden1, n_hidden2, n_output):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, noutput)
        self.initial_state_matrix = torch.nn.Embedding(node_num, nfeat).double()
        self.dropout1 = dropout1
        self.predict1 = torch.nn.Linear(n_feature, n_hidden1)
        self.predict2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.predict3 = torch.nn.Linear(n_hidden2, n_output)

    def forward1(self, lap_matrix_p, lap_matrix_n, incidence_matrix_preoder):
        x0 = self.initial_state_matrix.weight
        x1 = (self.gc1(x0, lap_matrix_p, lap_matrix_n, incidence_matrix_preoder))
        x1 = F.dropout(x1, self.dropout1, training=self.training)
        x2 = (self.gc2(x1, lap_matrix_p, lap_matrix_n, incidence_matrix_preoder))
        return x2

    def forward2(self, x):
        x = F.relu(self.predict1(x))
        x = F.relu(self.predict2(x))
        x = self.predict3(x)
        return x

