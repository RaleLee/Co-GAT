import os
import sys
import json
import torch
import argparse

from utils import DataHub
from nn import TaggingAgent
from utils import fix_random_state
from utils import training, evaluate
from utils.dict import PieceAlphabet


parser = argparse.ArgumentParser()
# Pre-train Hyper parameter
parser.add_argument("--pretrained_model", "-pm", type=str, default="none",
                    choices=["none", "bert", "roberta", "xlnet", "albert", "electra"],
                    help="choose pretrained model, default is none.")
parser.add_argument("--linear_decoder", "-ld", action="store_true", default=False,
                    help="Using Linear decoder to get category.")
parser.add_argument("--bert_learning_rate", "-blr", type=float, default=1e-5,
                    help="The learning rate of all types of pretrain model.")
# Basic Hyper parameter
parser.add_argument("--data_dir", "-dd", type=str, default="dataset/mastodon")
parser.add_argument("--save_dir", "-sd", type=str, default="./save")
parser.add_argument("--batch_size", "-bs", type=int, default=16)
parser.add_argument("--num_epoch", "-ne", type=int, default=300)
parser.add_argument("--random_state", "-rs", type=int, default=0)

# Model Hyper parameter
parser.add_argument("--num_layer", "-nl", type=int, default=2,
                    help="This parameter CAN NOT be modified! Please use gat_layer to set the layer num of gat")
parser.add_argument("--gat_layer", "-gl", type=int, default=2,
                    help="Control the number of GAT layers. Must be between 2 and 4.")
parser.add_argument("--embedding_dim", "-ed", type=int, default=128)
parser.add_argument("--hidden_dim", "-hd", type=int, default=256)
parser.add_argument("--dropout_rate", "-dr", type=float, default=0.1)
parser.add_argument("--gat_dropout_rate", "-gdr", type=float, default=0.1)

args = parser.parse_args()
print(json.dumps(args.__dict__, indent=True), end="\n\n\n")

# fix random seed
fix_random_state(args.random_state)

# Build dataset
data_house = DataHub.from_dir_addadj(args.data_dir)
# piece vocab
piece_vocab = PieceAlphabet("piece", pretrained_model=args.pretrained_model)
model = TaggingAgent(
    data_house.word_vocab, piece_vocab, data_house.sent_vocab,
    data_house.act_vocab, data_house.adj_vocab, data_house.adj_full_vocab, data_house.adj_id_vocab, args.embedding_dim,
    args.hidden_dim, args.num_layer, args.gat_layer, args.gat_dropout_rate, args.dropout_rate,
    args.linear_decoder, args.pretrained_model)
if torch.cuda.is_available():
    model = model.cuda()
if args.data_dir == "dataset/mastodon":
    metric = False
else:
    metric = True
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

dev_best_sent, dev_best_act = 0.0, 0.0
test_sent_sent, test_sent_act = 0.0, 0.0
test_act_sent, test_act_act = 0.0, 0.0

for epoch in range(0, args.num_epoch + 1):
    print("Training Epoch: {:4d} ...".format(epoch), file=sys.stderr)

    train_loss, train_time = training(model, data_house.get_iterator("train", args.batch_size, True),
                                      10.0, args.bert_learning_rate, args.pretrained_model)
    print("[Epoch{:4d}], train loss is {:.4f}, cost {:.4f} s.".format(epoch, train_loss, train_time))

    dev_sent_f1, _, _, dev_act_f1, _, _, dev_time = evaluate(
        model, data_house.get_iterator("dev", args.batch_size, False), metric)
    test_sent_f1, sent_r, sent_p, test_act_f1, act_r, act_p, test_time = evaluate(
        model, data_house.get_iterator("test", args.batch_size, False), metric)

    print("On dev, sentiment f1: {:.4f}, act f1: {:.4f}".format(dev_sent_f1, dev_act_f1))
    print("On test, sentiment f1: {:.4f}, act f1 {:.4f}".format(test_sent_f1, test_act_f1))
    print("Dev and test cost {:.4f} s.\n".format(dev_time + test_time))

    # if get a higher score on dev set, do predict on test set, base on sent or act.
    if dev_sent_f1 > dev_best_sent or dev_act_f1 > dev_best_act:

        if dev_sent_f1 > dev_best_sent:
            dev_best_sent = dev_sent_f1

            test_sent_sent = test_sent_f1
            test_sent_act = test_act_f1

            print("<Epoch {:4d}>, Update (base on sent) test score: sentiment f1: {:.4f} (r: "
                  "{:.4f}, p: {:.4f}), act f1: {:.4f} (r: {:.4f}, p: {:.4f})"
                  ";".format(epoch, test_sent_sent, sent_r, sent_p, test_sent_act, act_r, act_p))

        if dev_act_f1 > dev_best_act:
            dev_best_act = dev_act_f1

            test_act_sent = test_sent_f1
            test_act_act = test_act_f1

            print("<Epoch {:4d}>, Update (base on act) test score: sentiment f1: {:.4f} (r: "
                  "{:.4f}, p: {:.4f}), act f1: {:.4f} (r: {:.4f}, p: {:.4f})"
                  ";".format(epoch, test_act_sent, sent_r, sent_p, test_act_act, act_r, act_p))

        torch.save(model, os.path.join(args.save_dir, "model.pt"))

        print("", end="\n")
