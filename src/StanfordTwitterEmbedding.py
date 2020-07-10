# https://nlp.stanford.edu/projects/glove/

from pathlib import Path
import torch
import torch.nn as nn

from Indexer import Indexer


class StanfordTwitterEmbedding(nn.Module):
    def __init__(self, device):
        super(StanfordTwitterEmbedding, self).__init__()
        self.path = Path('../data/models/glove.twitter.27B/glove.twitter.27B.200d.txt')
        with self.path.open('r', encoding='utf-8-sig') as f:
            texts = f.readlines()
        headers = [len(texts), None]
        contents = [text.split(' ') for text in texts]
        vocab = [content[0] for content in contents]
        weights = [list(map(float, content[1:])) for content in contents]
        self.indexer = Indexer(with_preprocess=False)
        for word in vocab:
            self.indexer.count_word(word)
            self.indexer.add_word(word)
        self.embedding_dim = len(weights[0])
        special_weights = [[0.0] * self.embedding_dim] * 5
        weights = torch.FloatTensor(special_weights + weights)
        self.embedding = nn.Embedding.from_pretrained(embeddings=weights, padding_idx=self.indexer.padding_index)
        self.device = device
        self.embedding.to(device)

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


if __name__ == '__main__':
    ne = StanfordTwitterEmbedding(device='cpu')
