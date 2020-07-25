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
        self.kernel_size = self.hyper_params['kernel_size']
        self.window_sizes = [self.hyper_params['window_size1'], self.hyper_params['window_size2'], self.hyper_params['window_size3']]
        self.hidden_size = emb_dim
        self.padding_sizes = [min(item // 2, 2) for item in self.window_sizes]
        self.input_chs = [1] * 3
        self.output_chs = [self.kernel_size] * 3

        self.dropout = nn.Dropout(hyper_params['dropout_ratio'])
        self.cnns = nn.ModuleList([nn.Conv1d(in_channels=self.hidden_size, out_channels=och, kernel_size=(ws,), stride=1, padding=pd) for ich, och, ws, pd in zip(self.input_chs, self.output_chs, self.window_sizes, self.padding_sizes)])

        self.num_head = 5
        self.num_layer = 6
        # self.attentions = nn.ModuleList([nn.ModuleList([Attention(dimensions=self.kernel_size // self.num_head) for _ in range(self.num_head)]) for _ in range(self.num_layer)])
        self.attentions = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.kernel_size, num_heads=self.num_head, dropout=hyper_params['dropout_ratio']) for _ in range(self.num_layer)])
        self.linears = nn.ModuleList([nn.Linear(self.kernel_size, self.kernel_size) for _ in range(self.num_layer)])

        self.output = nn.Linear(self.hidden_size * self.hidden_size, 1)

        self.to(device)

    def forward(self, batch_sentence):
        embeddings = [embedding(batch_sentence) for embedding in self.embeddings]
        embeddings = torch.cat(embeddings, dim=2)
        batch_size = embeddings.shape[0]
        max_len = embeddings.shape[1]

        # feature_map = [torch.max_pool2d(torch.relu(layer(self.dropout(embeddings.unsqueeze(1)))), (max_len)).squeeze() for layer in self.cnns]
        # feature_map = [torch.max_pool1d(torch.relu(layer(self.dropout(embeddings.unsqueeze(1)))).squeeze(), kernel_size=max_len).squeeze() for layer in self.cnns]
        feature_map = [torch.relu(layer(self.dropout(embeddings.transpose(2, 1)))) for layer in self.cnns]
        l = torch.cat(feature_map, dim=2)
        # pooled_multi_feature_maps = F.max_pool2d(multi_feature_maps, kernel_size=(max_len, 1)).squeeze(2)
        l = l.transpose(2, 1)

        seq_logits, attention_weights = [], []
        for i in range(self.num_layer):
            atten_l, w = self.attentions[i](query=l, key=l, value=l)
            l = l + atten_l
            l = torch.layer_norm(l, normalized_shape=(l.shape[1], l.shape[2]))
            fc_l = self.linears[i](l)
            l = l + fc_l
            l = torch.layer_norm(l, normalized_shape=(l.shape[1], l.shape[2]))
            seq_logits.append(l)
            attention_weights.append(w)

        refined_seq = seq_logits[-1].view(batch_size, -1).clone()
        if refined_seq.shape[1] > self.output.in_features:
            refined_seq = refined_seq.narrow(1, 0, self.output.in_features)
        elif refined_seq.shape[1] < self.output.in_features:
            refined_seq = torch.cat((refined_seq, torch.zeros(batch_size, self.output.in_features - refined_seq.shape[1])), dim=1)

        output = self.output(refined_seq)
        return output


'''
'''

