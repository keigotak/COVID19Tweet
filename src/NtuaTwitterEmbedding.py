from pathlib import Path
import torch
import torch.nn as nn

from Indexer import Indexer


class NtuaTwitterEmbedding(nn.Module):
    def __init__(self, device, stop_words=None):
        super(NtuaTwitterEmbedding, self).__init__()
        self.path = Path('../data/models/ntua-slp-semeval2018/ntua_twitter_300.txt')
        with self.path.open('r', encoding='utf-8-sig') as f:
            texts = f.readlines()
        headers = texts[0].strip().split(' ')
        contents = [text.strip().split(' ') for text in texts[1:]]
        vocab = [content[0] for content in contents]
        weights = [list(map(float, content[1:])) for content in contents]
        self.indexer = Indexer(special_tokens={'<s>': 0, '<unk>': 1, '<pad>': 2, '<\s>': 3, '<mask>': 4}, with_preprocess=False, stop_words=stop_words)
        for word in vocab:
            self.indexer.count_word(word)
            self.indexer.add_word(word)
        self.embedding_dim = int(headers[1])
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
    ne = NtuaTwitterEmbedding(device='cpu')