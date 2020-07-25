import torch
import torch.nn as nn

from AbstractModel import AbstractModel
from Attention import Attention


class Cnn(AbstractModel):
    def __init__(self, device='cpu', hyper_params=None):
        sup = super()
        sup.__init__(device=device, hyper_params=hyper_params)
        self.embeddings = nn.ModuleList([sup.get_embeddings(key=key, device=device) for key in self.hyper_params['embeddings']])

        emb_dim = sum([item.embedding_dim for item in self.embeddings])
        self.hidden_size = emb_dim
        self.kernel_size = self.hyper_params['kernel_size']
        self.window_sizes = [self.hyper_params['window_size1'], self.hyper_params['window_size2'], self.hyper_params['window_size3']]
        self.input_chs = [1, 1, 1]
        self.output_chs = [self.kernel_size, self.kernel_size, self.kernel_size]

        self.dropout = nn.Dropout(hyper_params['dropout_ratio'])
        self.cnns = nn.ModuleList([nn.Conv2d(in_channels=ich, out_channels=och, kernel_size=(ws, emb_dim), stride=1) for ich, och, ws in zip(self.input_chs, self.output_chs, self.window_sizes)])

        self.num_head = hyper_params['num_head']
        self.attention = nn.ModuleList([Attention(dimensions=self.kernel_size) for _ in range(self.num_head)])

        self.output = nn.Linear(sum(self.output_chs), 1)

        self.to(device)

    def forward(self, batch_sentence):
        embeddings = [embedding(batch_sentence) for embedding in self.embeddings]
        embeddings = torch.cat(embeddings, dim=2)
        max_len = embeddings.shape[1] - 2

        # feature_map = [torch.max_pool2d(torch.relu(layer(self.dropout(embeddings.unsqueeze(1)))), (max_len)).squeeze() for layer in self.cnns]
        # feature_map = [torch.max_pool1d(torch.relu(layer(self.dropout(embeddings.unsqueeze(1)))).squeeze(), kernel_size=max_len).squeeze() for layer in self.cnns]
        feature_map = [torch.relu(layer(self.dropout(embeddings.unsqueeze(1)))).squeeze() for layer in self.cnns]
        multi_feature_maps = torch.cat(feature_map, dim=2)

        seq_logits, attention_weights = [], []
        for i in range(self.num_head):
            l, w = self.attention[i](query=multi_feature_maps, context=multi_feature_maps)
            seq_logits.append(l)
            attention_weights.append(w)

        output = self.output(multi_feature_maps)
        return output


'''
'''

