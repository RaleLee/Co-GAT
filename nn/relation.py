import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphRelation(nn.Module):
    """
    Deep Co-Interactive Graph Layer
    """

    def __init__(self, hidden_dim, dropout_rate, n_layer=2):
        super(GraphRelation, self).__init__()

        self._sent_linear = nn.Linear(
            hidden_dim, hidden_dim, bias=False
        )
        self._act_linear = nn.Linear(
            hidden_dim, hidden_dim, bias=False
        )
        self._dialog_layer = GAT(hidden_dim, hidden_dim, hidden_dim, dropout_rate, 0.2, 8, n_layer)

    # Add for loading best model
    def add_missing_arg(self, layer=2):
        self._dialog_layer.add_missing_arg(layer)

    def forward(self, input_s, input_a, len_list, adj_re):
        graph_input = torch.cat([input_s, input_a], dim=1)
        ret = self._dialog_layer(graph_input, adj_re)
        # chunk into sent and act representation
        sent, act = torch.chunk(ret, 2, dim=1)
        return sent, act


class GAT(nn.Module):
    """
    multi-GAT layer stack version
    Thanks to https://github.com/Diego999/pyGAT
    """
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, layer=2):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        # stack GAT layer here
        # firstlayer and second layer will be initialized.
        # But will not be used if layer<3
        self.firstlayer = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=False)
        self.secondlayer = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=False)
        self.layer = layer

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    # Add for loading best model
    def add_missing_arg(self, layer=2):
        self.layer = layer

    def forward(self, x, adj):
        input_x = x
        x = F.dropout(x, self.dropout, training=self.training)
        # Only can accept up to 4 layers
        # Deep network is difficult to obtain better results
        if self.layer >= 3:
            x = self.firstlayer(x, adj)
            x = F.dropout(x, self.dropout, training=self.training)
        if self.layer == 4:
            x = self.secondlayer(x, adj)
        if self.layer > 4:
            assert False
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        # residual connection
        return x + input_x


class GraphAttentionLayer(nn.Module):
    """
    Thanks to https://github.com/Diego999/pyGAT
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2 * out_features, 1)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = self.W(input)
        N = h.size()[1]

        a_input = torch.cat([h.repeat(1, 1, N).view(h.shape[0], N * N, -1), h.repeat(1, N, 1)], dim=2)\
            .view(h.shape[0], N, -1, 2 * self.out_features)
        e = self.leakyrelu(self.a(a_input).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)

        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = []

        for per_a, per_h in zip(attention, h):
            h_prime.append(torch.matmul(per_a, per_h))

        h_prime = torch.stack(h_prime)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
