from transformers import BertModel, RobertaModel, XLNetModel, AlbertModel, ElectraModel

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiGraphEncoder(nn.Module):
    def __init__(self,
                 word_embedding: nn.Embedding,
                 hidden_dim: int,
                 dropout_rate: float,
                 pretrained_model: str):
        """
        Use BiLSTM + GAT to Encode
        """

        super(BiGraphEncoder, self).__init__()

        # self._remove_user = remove_user
        if pretrained_model != "none":
            self._utt_encoder = UtterancePretrainedModel(hidden_dim, pretrained_model)
        else:
            self._utt_encoder = BiRNNEncoder(word_embedding, hidden_dim, dropout_rate)
        self._pretrained_model = pretrained_model

        self._dialog_layer_user = GAT(hidden_dim, hidden_dim, hidden_dim, dropout_rate, 0.2, 8)

    # Add for loading best model
    def add_missing_arg(self, pretrained_model):
        self._pretrained_model = pretrained_model

    def forward(self, input_w, adj, adj_full, mask=None):
        if self._pretrained_model != "none":
            hidden_w = self._utt_encoder(input_w, mask)
        else:
            hidden_w = self._utt_encoder(input_w)
        bi_ret = hidden_w

        ret = self._dialog_layer_user(bi_ret, adj)
        return ret


class BiRNNEncoder(nn.Module):

    def __init__(self,
                 word_embedding: nn.Embedding,
                 hidden_dim: int,
                 dropout_rate: float):

        super(BiRNNEncoder, self).__init__()

        _, embedding_dim = word_embedding.weight.size()
        self._word_embedding = word_embedding

        self._rnn_cell = nn.LSTM(embedding_dim, hidden_dim // 2,
                                 batch_first=True, bidirectional=True)
        self._drop_layer = nn.Dropout(dropout_rate)

    def forward(self, input_w):
        embed_w = self._word_embedding(input_w)
        dropout_w = self._drop_layer(embed_w)

        hidden_list, batch_size = [], input_w.size(0)
        for index in range(0, batch_size):
            batch_w = dropout_w[index]
            encode_h, _ = self._rnn_cell(batch_w)

            pooling_h, _ = torch.max(encode_h, dim=-2)
            hidden_list.append(pooling_h.unsqueeze(0))

        # Concatenate the representations of each sentence in the batch.
        return torch.cat(hidden_list, dim=0)


class GAT(nn.Module):
    """
    Thanks to https://github.com/Diego999/pyGAT
    """
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        input_x = x
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


class UtterancePretrainedModel(nn.Module):
    HIDDEN_DIM = 768

    def __init__(self, hidden_dim, pretrained_model):
        super(UtterancePretrainedModel, self).__init__()
        self._pretrained_model = pretrained_model

        if pretrained_model == "bert":
            self._encoder = BertModel.from_pretrained("bert-base-uncased")
        elif pretrained_model == "roberta":
            self._encoder = RobertaModel.from_pretrained("roberta-base")
        elif pretrained_model == "xlnet":
            self._encoder = XLNetModel.from_pretrained("xlnet-base-cased")
        elif pretrained_model == "albert":
            self._encoder = AlbertModel.from_pretrained("albert-base-v2")
        elif pretrained_model == "electra":
            self._encoder = ElectraModel.from_pretrained("google/electra-base-discriminator")
        else:
            assert False, "Something wrong with the parameter --pretrained_model"

        self._linear = nn.Linear(UtterancePretrainedModel.HIDDEN_DIM, hidden_dim)

    def forward(self, input_p, mask):
        cls_list = []

        for idx in range(0, input_p.size(0)):
            if self._pretrained_model == "electra":
                cls_tensor = self._encoder(input_p[idx], attention_mask=mask[idx])[0]
            else:
                cls_tensor, _ = self._encoder(input_p[idx], attention_mask=mask[idx])
            cls_tensor = cls_tensor[:, 0, :]
            linear_out = self._linear(cls_tensor.unsqueeze(0))
            cls_list.append(linear_out)
        return torch.cat(cls_list, dim=0)
