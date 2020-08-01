# https://nlp.stanford.edu/projects/glove/

from pathlib import Path
import torch
import torch.nn as nn

from Indexer import Indexer
from AbstractEmbedding import AbstractEmbedding


class StanfordTwitterEmbedding(AbstractEmbedding):
    def __init__(self, device):
        super(StanfordTwitterEmbedding, self).__init__(device=device)
        self.path = Path('../data/models/glove.twitter.27B/glove.twitter.27B.200d.txt')
        with self.path.open('r', encoding='utf-8-sig') as f:
            texts = f.readlines()
        headers = [len(texts), None]
        contents = [text.split(' ') for text in texts]
        vocab = [content[0] for content in contents]
        weights = [list(map(float, content[1:])) for content in contents]
        self.indexer = Indexer(special_tokens={'<s>': 0, '<unk>': 1, '<pad>': 2, '<\s>': 3, '<mask>': 4}, with_del_stopwords=self.with_del_stopwords)
        for word in vocab:
            self.indexer.count_word(word)
            self.indexer.add_word(word)
        self.embedding_dim = len(weights[0])
        special_weights = [[0.0] * self.embedding_dim] * 5
        weights = torch.FloatTensor(special_weights + weights)
        self.embedding = nn.Embedding.from_pretrained(embeddings=weights, padding_idx=self.indexer.padding_index)
        self.embedding.to(device)


if __name__ == '__main__':
    ne = StanfordTwitterEmbedding(device='cpu')
