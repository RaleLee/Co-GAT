import json
import random
import codecs

import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import torch
from torch import cuda


def fix_random_state(seed_value):
    """
    fix the random seed of each library
    """

    random.seed(seed_value)
    np.random.seed(seed_value)

    if torch.cuda.is_available():
        cuda.manual_seed(seed_value)
        cuda.manual_seed_all(seed_value)
    torch.manual_seed(seed_value)
    torch.random.manual_seed(seed_value)


def noise_augment(word_vocab, token_list, base=1.0):
    unk_sign = word_vocab.UNK_SIGN

    replace_list = []
    for token in token_list:
        freq = word_vocab.get_freq(token)
        prob = 1.0 / (freq + base)

        if random.random() < prob:
            replace_list.append(unk_sign)
        else:
            replace_list.append(token)

    # return the alphabet with UNK.
    return replace_list


def iterable_support(func, query):
    """
    Strengthen func, support iterative mapping query.
    """

    if isinstance(query, (list, tuple)):
        return [iterable_support(func, e) for e in query]

    # Support function query and index query at the same time.
    try:
        return func(query)
    except TypeError:
        return func[query]


def expand_list(n_list):
    e_list = []
    for element in n_list:
        if isinstance(element, (list, tuple)):
            e_list.extend(expand_list(element))
        else:
            e_list.append(element)
    return e_list


def nest_list(e_list, len_list):
    n_list, sent_start = [], 0

    for i in range(0, len(len_list)):
        sent_end = sent_start + len_list[i]
        n_list.append(e_list[sent_start: sent_end])
        sent_start = sent_end
    return n_list


def load_json_file(file_path):
    with codecs.open(file_path, "r", "utf-8") as fr:
        data_set = json.load(fr)
    return data_set


def load_txt(file_path):
    with open(file_path, "r") as f:
        dataset = f.readlines()
    f.close()
    return dataset


class ReferMetric(object):
    """
    On Mastodon dataset, following Cerisara et al. (2018), we ignore the neural sentiment label
    and adopt the average of the dialog-act specific F1 scores weighted by the prevalence of each dialog act for DAR.
    Reference to https://github.com/cerisara/DialogSentimentMastodon/code/hierarchicalRNN/data.py method evalDA, evalSE.
    """

    def __init__(self,
                 num_emot: int,
                 num_act: int,
                 pos_emot: int,
                 neg_emot: int):
        """
        Only calculate the neg and pos label.
        """

        self._num_emot = num_emot
        self._num_act = num_act

        self._pos_emot = pos_emot
        self._neg_emot = neg_emot

    @staticmethod
    def _base_statistic(haty, goldy, nclasses):
        nok = [0.] * nclasses
        nrec = [0.] * nclasses
        ntot = [0.] * nclasses

        for i in range(len(haty)):
            recy = haty[i]
            gldy = goldy[i]
            ntot[gldy] += 1
            nrec[recy] += 1
            if recy == gldy:
                nok[gldy] += 1
        return nok, nrec, ntot

    def validate_act(self, haty, goldy):
        nok, nrec, ntot = self._base_statistic(haty, goldy, self._num_act)

        nsamps = sum(ntot)
        preval = [float(ntot[i]) / float(nsamps) for i in range(self._num_act)]
        prec = 0.
        reca = 0.
        for j in range(self._num_act):
            tp = nok[j]
            pr, re = 0., 0.
            if nrec[j] > 0:
                pr = float(tp) / float(nrec[j])
            if ntot[j] > 0:
                re = float(tp) / float(ntot[j])
            prec += pr * preval[j]
            reca += re * preval[j]
        if prec + reca == 0.:
            f1 = 0.
        else:
            f1 = 2. * prec * reca / (prec + reca)
        return f1, prec, reca

    def validate_emot(self, haty, goldy):
        nok, nrec, ntot = self._base_statistic(haty, goldy, self._num_emot)

        f1pos, f1neg = 0., 0.
        avg_r, avg_p = 0.0, 0.0

        for j in (self._pos_emot,):  # 1=+ and 2=-
            tp = nok[j]
            pr, re = 0., 0.
            if nrec[j] > 0:
                pr = float(tp) / float(nrec[j])
            if ntot[j] > 0:
                re = float(tp) / float(ntot[j])
            if pr + re > 0.:
                f1pos = 2. * pr * re / (pr + re)
            avg_r += re
            avg_p += pr
        for j in (self._neg_emot,):  # 1=+ and 2=-
            tp = nok[j]
            pr, re = 0., 0.
            if nrec[j] > 0:
                pr = float(tp) / float(nrec[j])
            if ntot[j] > 0:
                re = float(tp) / float(ntot[j])
            if pr + re > 0.:
                f1neg = 2. * pr * re / (pr + re)
            avg_r += re
            avg_p += pr
        f1 = (f1pos + f1neg) / 2.
        return f1, avg_r / 2.0, avg_p / 2.0


class NormalMetric(object):
    """
    Using sklearn f1 score.
    """

    @staticmethod
    def validate_act(pred_list, gold_list):
        flat_pred_list = expand_list(pred_list)
        flat_gold_list = expand_list(gold_list)

        f_score = f1_score(flat_gold_list, flat_pred_list, average="macro")
        r_score = recall_score(flat_gold_list, flat_pred_list, average="macro")
        p_score = precision_score(flat_gold_list, flat_pred_list, average="macro")

        # The score calculated with macro-f1 is generally lower.
        return f_score, r_score, p_score

    @staticmethod
    def validate_emot(pred_list, gold_list):
        flat_pred_list = expand_list(pred_list)
        flat_gold_list = expand_list(gold_list)

        f_score = f1_score(flat_gold_list, flat_pred_list, average="macro")
        r_score = recall_score(flat_gold_list, flat_pred_list, average="macro")
        p_score = precision_score(flat_gold_list, flat_pred_list, average="macro")
        return f_score, r_score, p_score
