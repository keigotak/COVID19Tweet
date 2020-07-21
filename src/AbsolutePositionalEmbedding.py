import torch
import torch.nn as nn

from Indexer import Indexer


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, device, stop_words=set(), tokenizer=None):
        super(AbsolutePositionalEmbedding, self).__init__()
        self.max_length = 150
        self.indexer = Indexer(special_tokens={'<s>': 0, '<unk>': 1, '<pad>': 2, '<\s>': 3, '<mask>': 4}, with_preprocess=True)
        self.indexer.add_sentence(list(map(str, range(self.max_length))), with_raw=True, stop_words=stop_words)
        self.embedding_dim = 20
        self.embedding = nn.Embedding(num_embeddings=len(self.indexer), embedding_dim=self.embedding_dim, padding_idx=self.indexer.padding_index)
        self.device = device
        self.embedding.to(device)

    def forward(self, sentences):
        sentences = [self.indexer.tokenize(sentence) for sentence in sentences]
        sentences = [[str(i) for i, _ in enumerate(sentence)] for sentence in sentences]
        indexes = self.indexer.text_to_index(sentences, with_raw=True)
        pad_indexes = self.pad_sequence(indexes)
        pad_indexes = torch.Tensor(pad_indexes).long().to(self.device)
        vectors = self.embedding(pad_indexes)
        return vectors

    def pad_sequence(self, indexes):
        max_len = max(map(len, indexes))
        pad_sequences = [sentence + [self.indexer.padding_index] * (max_len - len(sentence)) if len(sentence) < max_len else sentence for sentence in indexes]
        return pad_sequences


if __name__ == '__main__':
    ne = AbsolutePositionalEmbedding(device='cpu')