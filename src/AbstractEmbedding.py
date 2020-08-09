import torch
import torch.nn as nn


class WithDelStopwords:
    __singleton = None
    __with_del_stopwords = None

    def __new__(cls):
        if cls.__singleton is None:
            cls.__singleton = super(WithDelStopwords, cls).__new__(cls)
            cls.__with_del_stopwords = False
        return cls.__singleton

    def get_instance(self):
        return self.__with_del_stopwords


class AbstractEmbedding(nn.Module):
    def __init__(self, device):
        super(AbstractEmbedding, self).__init__()
        self.device = device
        self.with_del_stopwords = WithDelStopwords().get_instance()

    def forward(self, sentences):
        indexes = self.indexer.text_to_index(sentences)
        pad_indexes = self.pad_sequence(indexes)
        pad_indexes = torch.Tensor(pad_indexes).long().to(self.device)
        vectors = self.embedding(pad_indexes)
        return vectors

    def pad_sequence(self, indexes):
        max_len = max(map(len, indexes))
        pad_sequences = [sentence + [self.indexer.padding_index] * (max_len - len(sentence)) if len(sentence) < max_len else sentence for sentence in indexes]
        return pad_sequences
