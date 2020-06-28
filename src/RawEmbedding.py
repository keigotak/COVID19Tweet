import torch
import torch.nn as nn

from Indexer import Indexer


class RawEmbedding(nn.Module):
    def __init__(self):
        super(RawEmbedding, self).__init__()
        self.indexer = Indexer()
        from HelperFunctions import get_datasets
        datasets, tags = get_datasets()
        sentences = [pairs[0] for pairs in datasets['train']]
        self.indexer.count_word_in_text(sentences)
        self.indexer.add_sentences(sentences)
        self.embedding_dim = 100
        self.embedding = nn.Embeddings(num_embeddings=len(self.indexer), embedding_dim=self.embedding_dim, ignore_index=self.indexer.padding_index)

    def forward(self, sentences):
        indexes = self.indexer.text_to_index(sentences)
        vectors = self.embedding(indexes)
        return vectors


if __name__ == '__main__':
    ne = RawEmbedding()