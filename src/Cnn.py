import torch
import torch.nn as nn
import torch.nn.functional as F

from AbstractModel import AbstractModel


class Cnn(AbstractModel):
    def __init__(self, device='cpu', hyper_params=None):
        sup = super()
        sup.__init__(device=device, hyper_params=hyper_params)
        self.embeddings = nn.ModuleList([sup.get_embeddings(key=key, device=device) for key in self.hyper_params['embeddings']])

        emb_dim = sum([item.embedding_dim for item in self.embeddings])
        self.kernel_size = self.hyper_params['kernel_size']
        self.window_sizes = [self.hyper_params['window_size1'], self.hyper_params['window_size2'], self.hyper_params['window_size3']]
        self.hidden_size = emb_dim
        self.padding_sizes = [item // 2 for item in self.window_sizes]
        self.input_chs = [1] * 3
        self.output_chs = [self.kernel_size] * 3

        self.dropout = nn.Dropout(hyper_params['dropout_ratio'])
        self.cnns = nn.ModuleList([nn.Conv2d(in_channels=ich, out_channels=och, kernel_size=(ws, emb_dim), stride=1, padding=(pd, 0)) for ich, och, ws, pd in zip(self.input_chs, self.output_chs, self.window_sizes, self.padding_sizes)])

        self.output = nn.Linear(sum(self.output_chs), hyper_params['num_class'])

        self.to(device)

    def forward(self, batch_sentence):
        embeddings = [embedding(batch_sentence) for embedding in self.embeddings]
        embeddings = torch.cat(embeddings, dim=2)
        seq_len = embeddings.shape[1]

        feature_maps = [torch.relu(layer(self.dropout(embeddings.unsqueeze(1)))) for layer in self.cnns]
        feature_maps = [F.max_pool1d(feature_map.squeeze(3), kernel_size=seq_len).squeeze(2) for feature_map in feature_maps]
        drop_output = self.dropout(torch.cat(feature_maps, dim=1))

        output = self.output(drop_output)
        return output


'''
2020.08.03 22:07:13|epoch:   7|train loss: 322.05|valid loss: 149.15|valid f1: 79.838|valid precision: 76.505|valid recall: 83.475|valid accuracy: 80.100|valid tp: 394|valid fp: 121|valid fn: 78|valid tn: 407
'''

