import torch
import torch.nn as nn

from Indexer import Indexer


class RawEmbedding(nn.Module):
    def __init__(self, device):
        super(RawEmbedding, self).__init__()
        self.indexer = Indexer(special_tokens={'<s>': 0, '<unk>': 1, '<pad>': 2, '<\s>': 3, '<mask>': 4})
        from HelperFunctions import get_datasets
        datasets, tags = get_datasets()
        sentences = [pairs[0] for pairs in datasets['train']]
        self.indexer.count_word_in_text(sentences)
        self.indexer.add_sentences(sentences)
        self.embedding_dim = 100
        self.embedding = nn.Embedding(num_embeddings=len(self.indexer), embedding_dim=self.embedding_dim, padding_idx=self.indexer.padding_index)
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
    ne = RawEmbedding(device='cpu')