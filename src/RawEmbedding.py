import torch.nn as nn

from Indexer import Indexer
from AbstractEmbedding import AbstractEmbedding
from Dataset import Dataset


class RawEmbedding(AbstractEmbedding):
    def __init__(self, device):
        super(RawEmbedding, self).__init__(device=device)
        self.indexer = Indexer(special_tokens={'<s>': 0, '<unk>': 1, '<pad>': 2, '<\s>': 3, '<mask>': 4})
        datasets = Dataset().get_instance()
        sentences = [pairs[0] for pairs in datasets['train']]
        self.indexer.count_word_in_text(sentences)
        self.indexer.add_sentences(sentences)
        self.embedding_dim = 100
        self.embedding = nn.Embedding(num_embeddings=len(self.indexer), embedding_dim=self.embedding_dim, padding_idx=self.indexer.padding_index)
        self.embedding.to(device)


if __name__ == '__main__':
    ne = RawEmbedding(device='cpu')