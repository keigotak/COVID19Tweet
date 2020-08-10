import torch
import torch.nn as nn

from AbstractModel import AbstractModel


class Mlp(AbstractModel):
    def __init__(self, device='cpu', hyper_params=None):
        sup = super()
        sup.__init__(device=device, hyper_params=hyper_params)
        self.embeddings = nn.ModuleList([sup.get_embeddings(key=key, device=device) for key in self.hyper_params['embeddings']])

        emb_dim = sum([item.embedding_dim for item in self.embeddings])
        self.hidden_size = emb_dim
        self.max_sentence_length = 256

        self.in_dimensions = [self.max_sentence_length * emb_dim, 1024, 256]
        self.out_dimensions = [1024, 256, 64]
        self.dropout = nn.Dropout(hyper_params['dropout_ratio'])
        self.layers = nn.ModuleList([nn.Linear(ind, outd) for ind, outd in zip(self.in_dimensions, self.out_dimensions)])

        self.output = nn.Linear(self.out_dimensions[-1], hyper_params['num_class'])

        self.to(device)

    def forward(self, batch_sentence):
        embeddings = [embedding(batch_sentence) for embedding in self.embeddings]
        embeddings = torch.cat(embeddings, dim=2)
        if embeddings.shape[1] < self.max_sentence_length:
            embeddings = torch.cat((embeddings, torch.zeros(embeddings.shape[0], self.max_sentence_length - embeddings.shape[1], self.hidden_size).to(self.device)), dim=1)
        elif embeddings.shape[1] > self.max_sentence_length:
            embeddings = embeddings.narrow(1, 0, self.max_sentence_length)
        embeddings = embeddings.view(embeddings.shape[0], -1)

        feature_maps = embeddings
        for layer in self.layers:
            feature_maps = torch.relu(layer(self.dropout(feature_maps)))
        drop_output = self.dropout(feature_maps)
        output = self.output(drop_output)
        return output


'''
2020.08.09 12:31:51|epoch:   6|train loss: 225.33|valid loss: 202.55|valid f1: 73.435|valid precision: 66.495|valid recall: 81.992|valid accuracy: 72.000|valid tp: 387|valid fp: 195|valid fn: 85|valid tn: 333
'''

