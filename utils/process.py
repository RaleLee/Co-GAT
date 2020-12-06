import time
from tqdm import tqdm

import torch
from torch.optim import Adam

from utils.help import NormalMetric, ReferMetric
from utils.help import iterable_support, expand_list

from transformers import AdamW


def training(model, data_iter, max_grad=10.0, bert_lr=1e-5, pretrained_model="none"):
    model.train()

    # using pretrain model need to change optimizer (Adam -> AdamW).
    if pretrained_model != "none":
        optimizer = AdamW(model.parameters(), lr=bert_lr, correct_bias=False)
    else:
        optimizer = Adam(model.parameters(), weight_decay=1e-8)
    time_start, total_loss = time.time(), 0.0

    for data_batch in tqdm(data_iter, ncols=50):
        batch_loss = model.measure(*data_batch)
        total_loss += batch_loss.cpu().item()

        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad
        )
        optimizer.step()

    time_con = time.time() - time_start
    return total_loss, time_con


def evaluate(model, data_iter, normal_metric):
    model.eval()

    gold_sent, pred_sent = [], []
    gold_act, pred_act = [], []
    time_start = time.time()

    for utt, sent, act, adj, adj_full, adj_I in tqdm(data_iter, ncols=50):
        gold_sent.extend(sent)
        gold_act.extend(act)

        with torch.no_grad():
            p_sent, p_act = model.predict(utt, adj, adj_full, adj_I)
        pred_sent.extend(p_sent)
        pred_act.extend(p_act)

    if not normal_metric:
        reference = ReferMetric(
            len(model.sent_vocab), len(model.act_vocab),
            model.sent_vocab.index("+"), model.sent_vocab.index("-")
        )
    else:
        reference = NormalMetric()

    pred_sent = iterable_support(model.sent_vocab.index, pred_sent)
    gold_sent = iterable_support(model.sent_vocab.index, gold_sent)
    pred_act = iterable_support(model.act_vocab.index, pred_act)
    gold_act = iterable_support(model.act_vocab.index, gold_act)

    pred_sent = expand_list(pred_sent)
    gold_sent = expand_list(gold_sent)
    pred_act = expand_list(pred_act)
    gold_act = expand_list(gold_act)

    sent_f1, sent_r, sent_p = reference.validate_emot(pred_sent, gold_sent)
    act_f1, act_r, act_p = reference.validate_act(pred_act, gold_act)
    fine_grain = model.show_fine_grain_act_performance(pred_act, gold_act)

    time_con = time.time() - time_start
    return sent_f1, sent_r, sent_p, act_f1, act_r, act_p, fine_grain, time_con
