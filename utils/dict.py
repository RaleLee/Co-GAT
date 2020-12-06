import json
from abc import abstractmethod
from collections import Counter
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer, AlbertTokenizer, ElectraTokenizer


class AbstractAlphabet(object):
    """
    The parent class of the alphabet, which specifies the functions that an alphabet should provide.
    """

    PAD_SIGN, UNK_SIGN = "[PAD]", "[UNK]"

    def __init__(self, name):
        self._name = name

        # Declare the mapping for serialization and deserialization.
        self._idx_to_elem = {}
        self._elem_to_idx = {}

        # element set
        self._element_set = set()

        # frequency of each element
        self._freq_counter = Counter()

    @property
    def name(self):
        return self._name

    def get_freq(self, element):
        try:
            return self._freq_counter[element]
        except KeyError:
            return 0

    def get(self, index):
        return self._idx_to_elem[index]

    def add(self, element):
        """
        Add an element into alphabet, can be re-write.
        """

        if element not in self._element_set:
            self._elem_to_idx[element] = len(self._element_set)
            self._idx_to_elem[len(self._element_set)] = element
            self._element_set.add(element)

        # freq add 1
        self._freq_counter[element] += 1

    @abstractmethod
    def index(self, element):
        pass

    def __len__(self):
        return len(self._idx_to_elem)

    def __str__(self):
        represent = json.dumps(
            self._elem_to_idx, indent=True, ensure_ascii=False
        )
        return represent


class WordAlphabet(AbstractAlphabet):
    """
    Word alphabet.
    PAD -> 0, UNK -> 1.
    """

    def __init__(self, name):
        super(WordAlphabet, self).__init__(name)

        # Add PAD and UNK first
        self.add(AbstractAlphabet.PAD_SIGN)
        self.add(AbstractAlphabet.UNK_SIGN)

    def index(self, element):
        try:
            return self._elem_to_idx[element]
        except KeyError:
            return self._elem_to_idx[self.UNK_SIGN]


class LabelAlphabet(AbstractAlphabet):
    """
    simple label alphabet
    """

    def index(self, element):
        return self._elem_to_idx[element]


class PieceAlphabet(AbstractAlphabet):
    """
    BERT word piece
    """

    CLS_SIGN, SEP_SIGN = "[CLS]", "[SEP]"

    def __init__(self, name, pretrained_model):
        super(PieceAlphabet, self).__init__(name)

        if pretrained_model == "none":
            # Because the padding method will do piece padding, must initialize a segment here.
            # But the padding result will not be used when setting "none"
            self._segment = RobertaTokenizer.from_pretrained("roberta-base")
        elif pretrained_model == "bert":
            self._segment = BertTokenizer.from_pretrained("bert-base-uncased")
        elif pretrained_model == "roberta":
            self._segment = RobertaTokenizer.from_pretrained("roberta-base")
        elif pretrained_model == "xlnet":
            self._segment = XLNetTokenizer.from_pretrained("xlnet-base-cased")
        elif pretrained_model == "albert":
            self._segment = AlbertTokenizer.from_pretrained("albert-base-v2")
        elif pretrained_model == "electra":
            self._segment = ElectraTokenizer.from_pretrained("google/electra-base-discriminator")
        else:
            assert False, "Something wrong with the parameter --pretrained_model"

    def index(self, elem_list):
        return self._segment.convert_tokens_to_ids(elem_list)

    def tokenize(self, word_list):
        """
        Notice that before piece, DO NOT add [CLS], [SEP] or [PAD]. Otherwise it will go wrong.
        """

        piece_list = []
        for word in word_list:
            s_list = self._segment.tokenize(word)
            piece_list.extend(s_list)
        return piece_list
