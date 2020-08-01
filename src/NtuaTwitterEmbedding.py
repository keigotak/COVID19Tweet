from pathlib import Path
import torch
import torch.nn as nn

from Indexer import Indexer
from AbstractEmbedding import AbstractEmbedding


class NtuaTwitterEmbedding(AbstractEmbedding):
    def __init__(self, device):
        super(NtuaTwitterEmbedding, self).__init__(device=device)
        self.path = Path('../data/models/ntua-slp-semeval2018/ntua_twitter_300.txt')
        with self.path.open('r', encoding='utf-8-sig') as f:
            texts = f.readlines()
        headers = texts[0].strip().split(' ')
        contents = [text.strip().split(' ') for text in texts[1:]]
        vocab = [content[0] for content in contents]
        weights = [list(map(float, content[1:])) for content in contents]
        self.indexer = Indexer(special_tokens={'<s>': 0, '<unk>': 1, '<pad>': 2, '<\s>': 3, '<mask>': 4}, with_del_stopwords=self.with_del_stopwords)
        for word in vocab:
            self.indexer.count_word(word)
            self.indexer.add_word(word)
        self.embedding_dim = int(headers[1])
        special_weights = [[0.0] * self.embedding_dim] * 5
        weights = torch.FloatTensor(special_weights + weights)
        self.embedding = nn.Embedding.from_pretrained(embeddings=weights, padding_idx=self.indexer.padding_index)
        self.embedding.to(device)


if __name__ == '__main__':
    ne = NtuaTwitterEmbedding(device='cpu')