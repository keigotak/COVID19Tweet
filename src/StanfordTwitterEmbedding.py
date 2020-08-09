# https://nlp.stanford.edu/projects/glove/

from joblib import Parallel, delayed
from pathlib import Path
import pickle

import torch
import torch.nn as nn

from Indexer import Indexer
from AbstractEmbedding import AbstractEmbedding


class StanfordTwitterEmbedding(AbstractEmbedding):
    def __init__(self, device):
        super(StanfordTwitterEmbedding, self).__init__(device=device)
        self.path = Path('../data/models/glove.twitter.27B/glove.twitter.27B.200d.txt')
        with_raw_file = False
        if with_raw_file:
            with self.path.open('r', encoding='utf-8-sig') as f:
                texts = f.readlines()
            headers = [len(texts), None]
            vocab, weights = map(list, zip(*Parallel(n_jobs=10)([delayed(self.get_weights)(text) for text in texts])))
            with (self.path.parent / 'vocab.pkl').open('wb') as f:
                pickle.dump(vocab, f)
            with (self.path.parent / 'weights.pkl').open('wb') as f:
                pickle.dump(weights, f)
        else:
            with (self.path.parent / 'vocab.pkl').open('rb') as f:
                vocab = pickle.load(f)
            with (self.path.parent / 'weights.pkl').open('rb') as f:
                weights = pickle.load(f)

        self.indexer = Indexer(special_tokens={'<s>': 0, '<unk>': 1, '<pad>': 2, '<\s>': 3, '<mask>': 4}, with_del_stopwords=self.with_del_stopwords)
        for word in vocab:
            self.indexer.count_word(word)
            self.indexer.add_word(word)
        self.embedding_dim = len(weights[0])
        special_weights = [[0.0] * self.embedding_dim] * 5
        weights = torch.FloatTensor(special_weights + weights)
        self.embedding = nn.Embedding.from_pretrained(embeddings=weights, padding_idx=self.indexer.padding_index)
        self.embedding.to(device)

    def get_weights(self, text):
        content = text.split(' ')
        return content[0], list(map(float, content[1:]))


if __name__ == '__main__':
    ne = StanfordTwitterEmbedding(device='cpu')
