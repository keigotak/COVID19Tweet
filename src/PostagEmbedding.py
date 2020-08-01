import torch
import torch.nn as nn
import nltk

from Indexer import Indexer
from AbstractEmbedding import AbstractEmbedding
from Dataset import Dataset


class PostagEmbedding(AbstractEmbedding):
    def __init__(self, device):
        super(PostagEmbedding, self).__init__(device=device)
        self.indexer = Indexer(special_tokens={'<s>': 0, '<unk>': 1, '<pad>': 2, '<\s>': 3, '<mask>': 4}, with_del_stopwords=False) # postag embedding の場合だけ必ずFalse
        datasets = Dataset().get_instance()
        sentences = [nltk.pos_tag(self.indexer.tokenize(pairs[0])) for pairs in datasets['train']]
        sentences = [[pairs[1] for pairs in sentence] for sentence in sentences]
        for sentence in sentences:
            self.indexer.add_sentence(sentence, with_raw=True)
        self.embedding_dim = 10
        self.embedding = nn.Embedding(num_embeddings=len(self.indexer), embedding_dim=self.embedding_dim, padding_idx=self.indexer.padding_index)
        self.embedding.to(device)

    def forward(self, sentences):
        if self.with_del_stopwords:
            postags = [nltk.pos_tag(self.indexer.tokenize(sentence)) for sentence in sentences]
            sentences = [[pairs[0] for pairs in postag] for postag in postags]
            postags = [[pairs[1] for pairs in postag] for postag in postags]
            is_stopword = self.indexer.is_stopword(sentences)
            postags = [[tag for sw, tag in zip(stopword, postag) if sw != 1] for stopword, postag in zip(is_stopword, postags)]
        else:
            postags = [nltk.pos_tag(self.indexer.tokenize(sentence)) for sentence in sentences]
            postags = [[pairs[1] for pairs in postag] for postag in postags]
        indexes = [[self.indexer.get_index(tag) for tag in postag] for postag in postags]
        pad_indexes = self.pad_sequence(indexes)
        pad_indexes = torch.Tensor(pad_indexes).long().to(self.device)
        vectors = self.embedding(pad_indexes)
        return vectors


if __name__ == '__main__':
    emb = PostagEmbedding(device='cuda:0')
    emb.forward(['I have a pen, I have a apple.'])
