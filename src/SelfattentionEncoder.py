import torch
import torch.nn as nn

from AbstractModel import AbstractModel
from Attention import Attention


class SelfattentionEncoder(AbstractModel):
    def __init__(self, device='cpu', hyper_params=None):
        sup = super()
        sup.__init__(device=device, hyper_params=hyper_params)
        self.embeddings = nn.ModuleList([sup.get_embeddings(key=key, device=device) for key in self.hyper_params['embeddings']])

        self.hidden_size = sum([item.embedding_dim for item in self.embeddings])
        # self.kernel_size = self.hyper_params['kernel_size']
        self.num_head = self.hyper_params['num_head']
        self.num_layer = self.hyper_params['num_layer']
        self.dropout = nn.Dropout(hyper_params['dropout_ratio'])
        self.attentions = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.num_head, dropout=hyper_params['dropout_ratio']) for _ in range(self.num_layer)])
        self.linears = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.num_layer)])

        self.output = nn.Linear(self.hidden_size * self.hidden_size, 1)

        self.to(device)

    def forward(self, batch_sentence):
        embeddings = [embedding(batch_sentence) for embedding in self.embeddings]
        l = torch.cat(embeddings, dim=2)
        batch_size = embeddings.shape[0]

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

