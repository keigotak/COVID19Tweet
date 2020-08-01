import torch.nn as nn

from RawEmbedding import RawEmbedding
from NtuaTwitterEmbedding import NtuaTwitterEmbedding
from StanfordTwitterEmbedding import StanfordTwitterEmbedding
from AbsolutePositionalEmbedding import AbsolutePositionalEmbedding
from PostagEmbedding import PostagEmbedding


class AbstractModel(nn.Module):
    def __init__(self, device='cpu', hyper_params=None):
        super(AbstractModel, self).__init__()
        self.hyper_params = hyper_params
        self.device = device

    def forward(self, batch_sentence):
        pass

    def get_embeddings(self, key, device):
        if key == 'ntua':
            return NtuaTwitterEmbedding(device=device)
        elif key == 'stanford':
            return StanfordTwitterEmbedding(device=device)
        elif key == 'raw':
            return RawEmbedding(device=device)
        elif key == 'position':
            return AbsolutePositionalEmbedding(device=device)
        elif key == 'postag':
            return PostagEmbedding(device=device)


if __name__ == '__main__':
    am = AbstractModel()
    embs = []
    for key in ['stanford', 'position', 'postag']:
        embs.append(am.get_embeddings(key, device='cuda:2'))
    # sentences = ['I have a pen. I have an apple.']
    sentences = ['Official death toll from #covid19 in the United Kingdom is now GREATER than: Germany + Poland + Switzerland + Austria + Portugal + Greece + Sweden + Finland + Norway + Ireland... COMBINED. UK: 67.5 Million (233 dead) Above group: 185 Million (230 dead) HTTPURL',
                 'Dearest Mr. President @USER 1,169 coronavirus deaths in the US in 24 hours (?) Covid19 pandemic is an international crime from China - not a nature disasster! Please use your authorities to protect your people and world against China! #ChinaHasToCompensateAll',
                 'Latest Updates March 20 ⚠️5274 new cases and 38 new deaths in the United States Illinois: Governo Pritzker issues "stay at home" order for all residents New York: Governor Cuomo orders 100% of all non-essential workers to stay home Penns...Source ( /coronavirus/country/us/ )',
                 '真把公主不当干部 BREAKING: 21 people on Grand Princess cruise ship docked off the California coast tested positive for coronavirus, including 19 crew members and two passengers, Vice Pres. Mike Pence says. 24 people tested negative. HTTPURL HTTPURL']

    results = []
    for emb in embs:
        results.append(emb(sentences))
    import torch
    print(torch.cat(results, dim=2).shape)

