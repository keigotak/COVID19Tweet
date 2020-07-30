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
