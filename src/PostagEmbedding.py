import torch
import torch.nn as nn
import nltk

from Indexer import Indexer
from AbstractEmbedding import AbstractEmbedding
from Dataset import Dataset


class PostagEmbedding(AbstractEmbedding):
    def __init__(self, device):
        super(PostagEmbedding, self).__init__(device=device)
        self.indexer = Indexer(special_tokens={'<s>': 0, '<unk>': 1, '<pad>': 2, '<\s>': 3, '<mask>': 4}, with_preprocess=False)
        datasets = Dataset().get_instance()
        sentences = [nltk.pos_tag(self.indexer.text_processor(pairs[0])) for pairs in datasets['train']]
        sentences = [[pairs[1] for pairs in sentence] for sentence in sentences]
        for sentence in sentences:
            self.indexer.add_sentence(sentence, with_raw=True)
        self.embedding_dim = 100
        self.embedding = nn.Embedding(num_embeddings=len(self.indexer), embedding_dim=self.embedding_dim, padding_idx=self.indexer.padding_index)
        self.embedding.to(device)

    def forward(self, sentences):
        sentences = [nltk.pos_tag(self.indexer.text_processor(sentence)) for sentence in sentences]
        sentences = [[pairs[1] for pairs in sentence] for sentence in sentences]
        indexes = [[self.indexer.get_index(word) for word in sentence] for sentence in sentences]
        pad_indexes = self.pad_sequence(indexes)
        pad_indexes = torch.Tensor(pad_indexes).long().to(self.device)
        vectors = self.embedding(pad_indexes)
        return vectors


if __name__ == '__main__':
    emb = PostagEmbedding(device='cuda:0')
    emb.forward(['I have a pen, I have a apple.'])
