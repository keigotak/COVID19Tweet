import torch.nn as nn

from Indexer import Indexer
from AbstractEmbedding import AbstractEmbedding
from Dataset import Dataset


class RawEmbedding(AbstractEmbedding):
    def __init__(self, device):
        super(RawEmbedding, self).__init__(device=device)
        self.indexer = Indexer(special_tokens={'<s>': 0, '<unk>': 1, '<pad>': 2, '<\s>': 3, '<mask>': 4}, with_del_stopwords=self.with_del_stopwords)
        datasets = Dataset().get_instance()
        sentences = [pairs[0] for pairs in datasets['train']]
        self.indexer.count_word_in_text(sentences)
        self.indexer.add_sentences(sentences)
        self.embedding_dim = 100
        self.embedding = nn.Embedding(num_embeddings=len(self.indexer), embedding_dim=self.embedding_dim, padding_idx=self.indexer.padding_index)
        self.embedding.to(device)


if __name__ == '__main__':
    ne = RawEmbedding(device='cpu')
    sentences = ['Official death toll from #covid19 in the United Kingdom is now GREATER than: Germany + Poland + Switzerland + Austria + Portugal + Greece + Sweden + Finland + Norway + Ireland... COMBINED. UK: 67.5 Million (233 dead) Above group: 185 Million (230 dead) HTTPURL',
                 'Dearest Mr. President @USER 1,169 coronavirus deaths in the US in 24 hours (?) Covid19 pandemic is an international crime from China - not a nature disasster! Please use your authorities to protect your people and world against China! #ChinaHasToCompensateAll',
                 'Latest Updates March 20 ⚠️5274 new cases and 38 new deaths in the United States Illinois: Governo Pritzker issues "stay at home" order for all residents New York: Governor Cuomo orders 100% of all non-essential workers to stay home Penns...Source ( /coronavirus/country/us/ )',
                 '真把公主不当干部 BREAKING: 21 people on Grand Princess cruise ship docked off the California coast tested positive for coronavirus, including 19 crew members and two passengers, Vice Pres. Mike Pence says. 24 people tested negative. HTTPURL HTTPURL']
    ret = ne(sentences)
    print(ret.shape)