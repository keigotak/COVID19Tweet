import torch
import torch.nn as nn

from Indexer import Indexer
from AbstractEmbedding import AbstractEmbedding


class AbsolutePositionalEmbedding(AbstractEmbedding):
    def __init__(self, device):
        super(AbsolutePositionalEmbedding, self).__init__(device=device)
        self.max_length = 150
        self.indexer = Indexer(special_tokens={'<s>': 0, '<unk>': 1, '<pad>': 2, '<\s>': 3, '<mask>': 4}, with_del_stopwords=self.with_del_stopwords)
        self.indexer.add_sentence(list(map(str, range(self.max_length))), with_raw=True)
        self.embedding_dim = 20
        self.embedding = nn.Embedding(num_embeddings=len(self.indexer), embedding_dim=self.embedding_dim, padding_idx=self.indexer.padding_index)
        self.embedding.to(device)

    def forward(self, sentences):
        sentences = [self.indexer.tokenize(sentence) for sentence in sentences]
        sentences = [[str(i) for i, _ in enumerate(sentence)] for sentence in sentences]
        indexes = [[self.indexer.get_index(word) for word in sentence] for sentence in sentences]
        pad_indexes = self.pad_sequence(indexes)
        pad_indexes = torch.Tensor(pad_indexes).long().to(self.device)
        vectors = self.embedding(pad_indexes)
        return vectors


if __name__ == '__main__':
    emb = AbsolutePositionalEmbedding(device='cpu')
    emb.forward(['I have a pen, I have a apple.'])
