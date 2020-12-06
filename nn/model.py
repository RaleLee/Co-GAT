import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.encode import BiGraphEncoder
from nn.decode import RelationDecoder, LinearDecoder

from utils.help import ReferMetric
from utils.dict import PieceAlphabet
from utils.load import WordAlphabet, LabelAlphabet
from utils.help import expand_list, noise_augment
from utils.help import nest_list, iterable_support


class TaggingAgent(nn.Module):

    def __init__(self,
                 word_vocab: WordAlphabet,
                 piece_vocab: PieceAlphabet,
                 sent_vocab: LabelAlphabet,
                 act_vocab: LabelAlphabet,
                 adj_vocab: LabelAlphabet,
                 adj_full_vocab: LabelAlphabet,
                 adj_id_vocab: LabelAlphabet,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layer: int,
                 gat_layer: int,
                 gat_dropout_rate: float,
                 dropout_rate: float,
                 use_linear_decoder: bool,
                 pretrained_model: str):

        super(TaggingAgent, self).__init__()

        self._piece_vocab = piece_vocab
        self._pretrained_model = pretrained_model

        self._word_vocab = word_vocab
        self._sent_vocab = sent_vocab
        self._act_vocab = act_vocab
        self._adj_vocab = adj_vocab
        self._adj_full_vocab = adj_full_vocab
        self._adj_id_vocab = adj_id_vocab

        self._encoder = BiGraphEncoder(
            nn.Embedding(len(word_vocab), embedding_dim),
            hidden_dim, dropout_rate, pretrained_model
        )
        if use_linear_decoder:
            self._decoder = LinearDecoder(len(sent_vocab), len(act_vocab), hidden_dim)
        else:
            self._decoder = RelationDecoder(
                len(sent_vocab), len(act_vocab), hidden_dim,
                num_layer, dropout_rate, gat_dropout_rate, gat_layer
            )

        # Loss function
        self._criterion = nn.NLLLoss(reduction="sum")

    # Add for loading best model
    def set_load_best_missing_arg(self, pretrained_model):
        self._pretrained_model = pretrained_model
        self._encoder.add_missing_arg(pretrained_model)
        self._decoder.add_missing_arg(2)

    def set_load_best_missing_arg_mastodon(self, pretrained_model, layer=2):
        self._pretrained_model = pretrained_model
        self._encoder.add_missing_arg(pretrained_model)
        self._decoder.add_missing_arg(layer)

    def forward(self, input_h, len_list, adj, adj_full, adj_re, mask=None):
        encode_h = self._encoder(input_h, adj, adj_full, mask)
        return self._decoder(encode_h, len_list, adj_re)

    @property
    def sent_vocab(self):
        return self._sent_vocab

    @property
    def act_vocab(self):
        return self._act_vocab

    def _wrap_padding(self, dial_list, adj_list, adj_full_list, adj_id_list, use_noise):
        dial_len_list = [len(d) for d in dial_list]
        max_dial_len = max(dial_len_list)

        adj_len_list = [len(adj) for adj in adj_list]
        max_adj_len = max(adj_len_list)

        # add adj_full
        adj_full_len_list = [len(adj_full) for adj_full in adj_full_list]
        max_adj_full_len = max(adj_full_len_list)

        # add adj_I
        adj_id_len_list = [len(adj_I) for adj_I in adj_id_list]
        max_adj_id_len = max(adj_id_len_list)

        assert max_dial_len == max_adj_len, str(max_dial_len) + " " + str(max_adj_len)
        assert max_adj_full_len == max_adj_len, str(max_adj_full_len) + " " + str(max_adj_len)
        assert max_adj_id_len == max_adj_full_len, str(max_adj_id_len) + " " + str(max_adj_full_len)

        turn_len_list = [[len(u) for u in d] for d in dial_list]
        max_turn_len = max(expand_list(turn_len_list))

        turn_adj_len_list = [[len(u) for u in adj] for adj in adj_list]
        max_turn_adj_len = max(expand_list(turn_adj_len_list))

        # add adj_full
        turn_adj_full_len_list = [[len(u) for u in adj_full] for adj_full in adj_full_list]
        max_turn_adj_full_len = max(expand_list(turn_adj_full_len_list))

        # add adj_I
        turn_adj_id_len_list = [[len(u) for u in adj_I] for adj_I in adj_id_list]
        max_turn_adj_id_len = max(expand_list(turn_adj_id_len_list))

        pad_adj_list = []
        for dial_i in range(0, len(adj_list)):
            pad_adj_list.append([])

            for turn in adj_list[dial_i]:
                pad_utt = turn + [0] * (max_turn_adj_len - len(turn))
                pad_adj_list[-1].append(pad_utt)

            if len(adj_list[dial_i]) < max_adj_len:
                pad_dial = [[0] * max_turn_adj_len] * (max_adj_len - len(adj_list[dial_i]))
                pad_adj_list[-1].extend(pad_dial)

        pad_adj_full_list = []
        for dial_i in range(0, len(adj_full_list)):
            pad_adj_full_list.append([])

            for turn in adj_full_list[dial_i]:
                pad_utt = turn + [0] * (max_turn_adj_full_len - len(turn))
                pad_adj_full_list[-1].append(pad_utt)

            if len(adj_full_list[dial_i]) < max_adj_full_len:
                pad_dial = [[0] * max_turn_adj_full_len] * (max_adj_full_len - len(adj_full_list[dial_i]))
                pad_adj_full_list[-1].extend(pad_dial)

        pad_adj_id_list = []
        for dial_i in range(0, len(adj_id_list)):
            pad_adj_id_list.append([])

            for turn in adj_id_list[dial_i]:
                pad_utt = turn + [0] * (max_turn_adj_id_len - len(turn))
                pad_adj_id_list[-1].append(pad_utt)

            if len(adj_id_list[dial_i]) < max_adj_id_len:
                pad_dial = [[0] * max_turn_adj_id_len] * (max_adj_id_len - len(adj_id_list[dial_i]))
                pad_adj_id_list[-1].extend(pad_dial)

        pad_adj_R_list = []
        for dial_i in range(0, len(pad_adj_id_list)):
            pad_adj_R_list.append([])
            assert len(pad_adj_id_list[dial_i]) == len(pad_adj_full_list[dial_i])
            for i in range(len(pad_adj_full_list[dial_i])):
                full = pad_adj_full_list[dial_i][i]
                # identify = pad_adj_id_list[dial_i][i]
                # if self._remove_history:
                #     pad_utt_up = identify + full
                # elif self._remove_interaction:
                #     pad_utt_up = full + identify
                # else:
                pad_utt_up = full + full
                pad_adj_R_list[-1].append(pad_utt_up)

            for i in range(len(pad_adj_full_list[dial_i])):
                full = pad_adj_full_list[dial_i][i]
                pad_utt_down = full + full
                pad_adj_R_list[-1].append(pad_utt_down)

        assert len(pad_adj_id_list[0]) * 2 == len(pad_adj_R_list[0]), pad_adj_R_list[0]

        pad_w_list, pad_sign = [], self._word_vocab.PAD_SIGN
        for dial_i in range(0, len(dial_list)):
            pad_w_list.append([])

            for turn in dial_list[dial_i]:
                if use_noise:
                    noise_turn = noise_augment(self._word_vocab, turn, 5.0)
                else:
                    noise_turn = turn
                pad_utt = noise_turn + [pad_sign] * (max_turn_len - len(turn))
                pad_w_list[-1].append(iterable_support(self._word_vocab.index, pad_utt))

            if len(dial_list[dial_i]) < max_dial_len:
                pad_dial = [[pad_sign] * max_turn_len] * (max_dial_len - len(dial_list[dial_i]))
                pad_w_list[-1].extend(iterable_support(self._word_vocab.index, pad_dial))

        cls_sign = self._piece_vocab.CLS_SIGN
        piece_list, sep_sign = [], self._piece_vocab.SEP_SIGN

        for dial_i in range(0, len(dial_list)):
            piece_list.append([])

            for turn in dial_list[dial_i]:
                seg_list = self._piece_vocab.tokenize(turn)
                piece_list[-1].append([cls_sign] + seg_list + [sep_sign])

            if len(dial_list[dial_i]) < max_dial_len:
                pad_dial = [[cls_sign, sep_sign]] * (max_dial_len - len(dial_list[dial_i]))
                piece_list[-1].extend(pad_dial)

        p_len_list = [[len(u) for u in d] for d in piece_list]
        max_p_len = max(expand_list(p_len_list))

        pad_p_list, mask = [], []
        for dial_i in range(0, len(piece_list)):
            pad_p_list.append([])
            mask.append([])

            for turn in piece_list[dial_i]:
                pad_t = turn + [pad_sign] * (max_p_len - len(turn))
                pad_p_list[-1].append(self._piece_vocab.index(pad_t))
                mask[-1].append([1] * len(turn) + [0] * (max_p_len - len(turn)))

        var_w_dial = torch.LongTensor(pad_w_list)
        var_p_dial = torch.LongTensor(pad_p_list)
        var_mask = torch.LongTensor(mask)
        var_adj_dial = torch.LongTensor(pad_adj_list)
        var_adj_full_dial = torch.LongTensor(pad_adj_full_list)
        var_adj_R_dial = torch.LongTensor(pad_adj_R_list)

        if torch.cuda.is_available():
            var_w_dial = var_w_dial.cuda()
            var_p_dial = var_p_dial.cuda()
            var_mask = var_mask.cuda()
            var_adj_dial = var_adj_dial.cuda()
            var_adj_full_dial = var_adj_full_dial.cuda()
            var_adj_R_dial = var_adj_R_dial.cuda()

        return var_w_dial, var_p_dial, var_mask, turn_len_list, p_len_list, var_adj_dial, var_adj_full_dial, \
            var_adj_R_dial

    def predict(self, utt_list, adj_list, adj_full_list, adj_id_list):
        var_utt, var_p, mask, len_list, _, var_adj, var_adj_full, var_adj_R = \
            self._wrap_padding(utt_list, adj_list, adj_full_list, adj_id_list, False)
        if self._pretrained_model != "none":
            pred_sent, pred_act = self.forward(var_p, len_list, var_adj, var_adj_full, var_adj_R, mask)
        else:
            pred_sent, pred_act = self.forward(var_utt, len_list, var_adj, var_adj_full, var_adj_R, None)

        trim_list = [len(l) for l in len_list]
        flat_sent = torch.cat(
            [pred_sent[i, :trim_list[i], :] for
             i in range(0, len(trim_list))], dim=0
        )
        flat_act = torch.cat(
            [pred_act[i, :trim_list[i], :] for
             i in range(0, len(trim_list))], dim=0
        )

        _, top_sent = flat_sent.topk(1, dim=-1)
        _, top_act = flat_act.topk(1, dim=-1)

        sent_list = top_sent.cpu().numpy().flatten().tolist()
        act_list = top_act.cpu().numpy().flatten().tolist()

        nest_sent = nest_list(sent_list, trim_list)
        nest_act = nest_list(act_list, trim_list)

        string_sent = iterable_support(
            self._sent_vocab.get, nest_sent
        )
        string_act = iterable_support(
            self._act_vocab.get, nest_act
        )
        return string_sent, string_act

    def measure(self, utt_list, sent_list, act_list, adj_list, adj_full_list, adj_id_list):
        var_utt, var_p, mask, len_list, _, var_adj, var_adj_full, var_adj_R = \
            self._wrap_padding(utt_list, adj_list, adj_full_list, adj_id_list, True)

        flat_sent = iterable_support(
            self._sent_vocab.index, sent_list
        )
        flat_act = iterable_support(
            self._act_vocab.index, act_list
        )

        index_sent = expand_list(flat_sent)
        index_act = expand_list(flat_act)

        var_sent = torch.LongTensor(index_sent)
        var_act = torch.LongTensor(index_act)
        if torch.cuda.is_available():
            var_sent = var_sent.cuda()
            var_act = var_act.cuda()

        if self._pretrained_model != "none":
            pred_sent, pred_act = self.forward(var_p, len_list, var_adj, var_adj_full, var_adj_R, mask)
        else:
            pred_sent, pred_act = self.forward(var_utt, len_list, var_adj, var_adj_full, var_adj_R, None)
        trim_list = [len(l) for l in len_list]

        flat_pred_s = torch.cat(
            [pred_sent[i, :trim_list[i], :] for
             i in range(0, len(trim_list))], dim=0
        )
        flat_pred_a = torch.cat(
            [pred_act[i, :trim_list[i], :] for
             i in range(0, len(trim_list))], dim=0
        )

        sent_loss = self._criterion(
            F.log_softmax(flat_pred_s, dim=-1), var_sent
        )
        act_loss = self._criterion(
            F.log_softmax(flat_pred_a, dim=-1), var_act
        )
        return sent_loss + act_loss

    def show_fine_grain_act_performance(self, pred_act, gold_act):
        result = ReferMetric.fine_grain_act_f1_table(pred_act, gold_act, self._act_vocab)
        return json.dumps(result, indent=True, ensure_ascii=True)
