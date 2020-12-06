import os
from copy import deepcopy

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.dict import WordAlphabet
from utils.dict import LabelAlphabet
from utils.help import load_json_file, load_txt
from utils.help import iterable_support


class DataHub(object):

    def __init__(self):
        self._word_vocab = WordAlphabet("word")

        self._sent_vocab = LabelAlphabet("sentiment")
        self._act_vocab = LabelAlphabet("act")
        self._adj_vocab = LabelAlphabet("adj")
        self._adj_full_vocab = LabelAlphabet("adj_full")
        self._adj_id_vocab = LabelAlphabet("adj_id")

        # using a dict to store the train, dev, test data
        self._data_collection = {}

    @property
    def word_vocab(self):
        return deepcopy(self._word_vocab)

    @property
    def sent_vocab(self):
        return deepcopy(self._sent_vocab)

    @property
    def act_vocab(self):
        return deepcopy(self._act_vocab)

    @property
    def adj_vocab(self):
        return deepcopy(self._adj_vocab)

    @property
    def adj_full_vocab(self):
        return deepcopy(self._adj_full_vocab)

    @property
    def adj_id_vocab(self):
        return deepcopy(self._adj_id_vocab)
        
    @classmethod
    def from_dir_addadj(cls, dir_path):
        house = DataHub()

        house._data_collection["train"] = house._read_data(
            os.path.join(dir_path, "train.json"), True
        )
        house._data_collection["dev"] = house._read_data(
            os.path.join(dir_path, "dev.json"), False
        )
        house._data_collection["test"] = house._read_data(
            os.path.join(dir_path, "test.json"), False
        ) 
        return house

    @staticmethod
    def _read_adj(file_path: str):
        lines = load_txt(file_path)
        adjs = []
        adjs_full_connect = []
        # Identity matrix
        adjs_I = []

        # Init adjacency matrix for each dialog
        for line in lines:
            if len(line) == 0:
                continue
            length = int(line.split()[0])
            edges = line.split()[1:]
            adj = [[0] * length for _ in range(length)]
            adj_I = [[0] * length for _ in range(length)]
            for i in range(length):
                adj_I[i][i] = 1
            adj_full = [[1] * length for _ in range(length)]
            for edge in edges:
                i, j = edge.split("-")
                adj[int(i)-1][int(j)-1] = 1
                adj[int(j)-1][int(i)-1] = 1
            adjs.append(adj)
            # full connect adj
            adjs_full_connect.append(adj_full)
            # Identity matrix
            adjs_I.append(adj_I)
        return adjs, adjs_full_connect, adjs_I

    def _read_data(self,
                   file_path: str,
                   build_vocab: bool = False):
        """
        On train, set build_vocab=True, will build alphabet
        """

        utt_list, sent_list, act_list = [], [], []
        dialogue_list = load_json_file(file_path)

        for session in dialogue_list:
            utt, emotion, act = [], [], []

            for interact in session:
                act.append(interact["act"])
                emotion.append(interact["sentiment"])

                word_list = interact["utterance"].split()
                utt.append(word_list)

            utt_list.append(utt)
            sent_list.append(emotion)
            act_list.append(act)

        adjpath = file_path.split(".")[0] + "_adj.txt"
        adj_list, adj_full_list, adj_I_list = self._read_adj(adjpath)
        
        if build_vocab:
            iterable_support(self._word_vocab.add, utt_list)
            iterable_support(self._sent_vocab.add, sent_list)
            iterable_support(self._act_vocab.add, act_list)
            iterable_support(self._adj_vocab.add, adj_list)
            iterable_support(self._adj_full_vocab.add, adj_full_list)
            iterable_support(self._adj_id_vocab.add, adj_I_list)

        # The returned list is based on dialogue, with three levels of nesting.
        return utt_list, sent_list, act_list, adj_list, adj_full_list, adj_I_list

    def get_iterator(self, data_name, batch_size, shuffle):
        data_set = _GeneralDataSet(*self._data_collection[data_name])

        data_loader = DataLoader(
            data_set, batch_size, shuffle, collate_fn=_collate_func
        )
        return data_loader


class _GeneralDataSet(Dataset):

    def __init__(self, utt, sent, act, adj, adj_full, adj_id):
        self._utt = utt
        self._sent = sent
        self._act = act
        self._adj = adj
        self._adj_full = adj_full
        self._adj_id = adj_id

    def __getitem__(self, item):
        return self._utt[item], self._sent[item], self._act[item],\
               self._adj[item], self._adj_full[item], self._adj_id[item]

    def __len__(self):
        return len(self._sent)


def _collate_func(instance_list):
    """
    As a function parameter to instantiate the DataLoader object.
    """

    n_entity = len(instance_list[0])
    scatter_b = [[] for _ in range(0, n_entity)]

    for idx in range(0, len(instance_list)):
        for jdx in range(0, n_entity):
            scatter_b[jdx].append(instance_list[idx][jdx])
    return scatter_b
