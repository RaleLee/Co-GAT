import torch.nn as nn

from nn.relation import GraphRelation


class RelationDecoder(nn.Module):

    def __init__(self,
                 num_sent: int,
                 num_act: int,
                 hidden_dim: int,
                 num_layer: int,
                 dropout_rate: float,
                 gat_dropout_rate: float,
                 gat_layer: int
                 ):
        super(RelationDecoder, self).__init__()
        self._num_layer = num_layer

        self._sent_layer_dict = nn.ModuleDict()
        self._act_layer_dict = nn.ModuleDict()

        # First with a BiLSTM layer to get the initial representation of SC and DAR
        self._sent_layer_dict.add_module(
                str(0), BiLSTMLayer(hidden_dim, dropout_rate)
            )
        self._act_layer_dict.add_module(
                str(0), BiLSTMLayer(hidden_dim, dropout_rate)
            )
        # After each calculation, the specified layer will be passed
        for layer_i in range(1, num_layer):
            self._sent_layer_dict.add_module(
                str(layer_i), UniLinearLayer(hidden_dim, dropout_rate)
            )
            self._act_layer_dict.add_module(
                str(layer_i), UniLSTMLayer(hidden_dim, dropout_rate)
            )

        self._relate_layer = GraphRelation(hidden_dim, gat_dropout_rate, n_layer=gat_layer)

        self._sent_linear = nn.Linear(hidden_dim, num_sent)
        self._act_linear = nn.Linear(hidden_dim, num_act)

    # Add for loading best model
    def add_missing_arg(self, layer=2):
        self._relate_layer.add_missing_arg(layer)

    def forward(self, input_h, len_list, adj_re):
        sent_h = self._sent_layer_dict["0"](input_h)
        act_h = self._act_layer_dict["0"](input_h)
        # residual connection
        res_s, res_a = sent_h + input_h, act_h + input_h
    
        sent_r, act_r = self._relate_layer(res_s, res_a, len_list, adj_re)

        # stack num layer CAN NOT be change here.
        # we ONLY stack 1 layer in our experiment.
        # We stack different GAT layer in relation.py
        # you can change gat_layer parameter to control the number of gat layer.
        sent_h = self._sent_layer_dict[str(1)](sent_r)
        act_h = self._act_layer_dict[str(1)](act_r)

        # residual connection
        sent_h, act_h = sent_h + input_h, act_h + input_h
        
        linear_s = self._sent_linear(sent_h)
        linear_a = self._act_linear(act_h)
        return linear_s, linear_a


class UniLSTMLayer(nn.Module):

    def __init__(self, hidden_dim, dropout_rate):
        super(UniLSTMLayer, self).__init__()

        self._rnn_layer = nn.LSTM(
            hidden_dim, hidden_size=hidden_dim,
            batch_first=True, bidirectional=False
        )
        self._drop_layer = nn.Dropout(dropout_rate)

    def forward(self, input_h):
        dropout_h = self._drop_layer(input_h)
        return self._rnn_layer(dropout_h)[0]


class BiLSTMLayer(nn.Module):

    def __init__(self, hidden_dim, dropout_rate):
        super(BiLSTMLayer, self).__init__()

        self._rnn_layer = nn.LSTM(
            hidden_dim, hidden_size=hidden_dim // 2,
            batch_first=True, bidirectional=True
        )
        self._drop_layer = nn.Dropout(dropout_rate)

    def forward(self, input_h):
        dropout_h = self._drop_layer(input_h)
        return self._rnn_layer(dropout_h)[0]


class UniLinearLayer(nn.Module):

    def __init__(self, hidden_dim, dropout_rate):
        super(UniLinearLayer, self).__init__()

        self._linear_layer = nn.Linear(hidden_dim, hidden_dim)
        self._drop_layer = nn.Dropout(dropout_rate)

    def forward(self, input_h):
        dropout_h = self._drop_layer(input_h)
        return self._linear_layer(dropout_h)


class LinearDecoder(nn.Module):
    def __init__(self, num_sent: int, num_act: int, hidden_dim: int):
        super(LinearDecoder, self).__init__()
        self._sent_linear = nn.Linear(hidden_dim, num_sent)
        self._act_linear = nn.Linear(hidden_dim, num_act)

    def forward(self, input_h, len_list, adj_re):
        return self._sent_linear(input_h), self._act_linear(input_h)
