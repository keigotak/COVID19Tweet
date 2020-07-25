import torch
import torch.nn as nn

from AbstractModel import AbstractModel
from Attention import Attention


class BiGruSelfattention(AbstractModel):
    def __init__(self, device='cpu', hyper_params=None):
        sup = super()
        sup.__init__(device=device, hyper_params=hyper_params)
        self.embeddings = nn.ModuleList([sup.get_embeddings(key=key, device=device) for key in self.hyper_params['embeddings']])

        emb_dim = sum([item.embedding_dim for item in self.embeddings])
        self.hidden_size = emb_dim
        self.f_gru1 = nn.GRU(input_size=emb_dim, hidden_size=emb_dim, batch_first=True)
        self.b_gru1 = nn.GRU(input_size=emb_dim, hidden_size=emb_dim, batch_first=True)
        self.f_gru2 = nn.GRU(input_size=emb_dim, hidden_size=emb_dim, batch_first=True)
        self.b_gru2 = nn.GRU(input_size=emb_dim, hidden_size=emb_dim, batch_first=True)

        self.num_head = hyper_params['num_head']
        self.attention = nn.ModuleList([Attention(dimensions=emb_dim) for _ in range(self.num_head)])

        self.dropout = nn.Dropout(hyper_params['dropout_ratio'])

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Linear(emb_dim, 1)

        self.to(device)

    def forward(self, batch_sentence):
        embeddings = [embedding(batch_sentence) for embedding in self.embeddings]
        embeddings = torch.cat(embeddings, dim=2)
        max_len = embeddings.shape[1]

        outf1, hidf1 = self.f_gru1(self.dropout(embeddings))
        resf1 = outf1 + embeddings
        rev_resf1 = resf1[:,torch.arange(max_len-1, -1, -1),:] # reversed

        outb1, hidb1 = self.b_gru1(self.dropout(rev_resf1))
        resb1 = outb1 + rev_resf1
        rev_resb1 = resb1[:,torch.arange(max_len-1, -1, -1),:] # not reversed

        outf2, hidf2 = self.f_gru2(self.dropout(rev_resb1))
        resf2 = outf2 + rev_resb1
        rev_resf2 = resf2[:,torch.arange(max_len-1, -1, -1),:] # reversed

        outb2, hidb2 = self.b_gru2(self.dropout(rev_resf2))
        resb2 = outb2 + rev_resf2
        rev_resb2 = resb2[:,torch.arange(max_len-1, -1, -1),:] # not reversed

        drop_output = self.dropout(rev_resb2)
        seq_logits, attention_weights = [], []
        for i in range(self.num_head):
            l, w = self.attention[i](query=drop_output, context=drop_output)
            seq_logits.append(l)
            attention_weights.append(w)
        avg_seq_logits = None
        for l in seq_logits:
            if avg_seq_logits is None:
                avg_seq_logits = l
            else:
                avg_seq_logits = avg_seq_logits + l
        avg_seq_logits = avg_seq_logits / self.num_head

        pooled_logits = self.pooling(avg_seq_logits.transpose(2, 1)).transpose(2, 1).squeeze()
        output = self.output(pooled_logits)
        return output


'''
 - with stanford twitter embedding 200d
2020.07.09 18:22:50|epoch:   9|train loss: 388.35|valid loss: 99.32|valid f1: 82.573|valid precision: 80.894|valid recall: 84.322|valid accuracy: 83.200|valid tp: 398|valid fp: 94|valid fn: 74|valid tn: 434

 - with stanford twitter embedding 100d
2020.07.09 15:38:28|epoch:   9|train loss: 496.90|valid loss: 103.71|valid f1: 81.206|valid precision: 77.247|valid recall: 85.593|valid accuracy: 81.300|valid tp: 404|valid fp: 119|valid fn: 68|valid tn: 409

 - with ntua twitter embedding
2020.07.09 14:18:49|epoch:  17|train loss: 311.18|valid loss: 102.94|valid f1: 83.452|valid precision: 80.117|valid recall: 87.076|valid accuracy: 83.700|valid tp: 411|valid fp: 102|valid fn: 61|valid tn: 426

 - apply ekphrasis
2020.07.09 02:02:37|epoch:  21|train loss: 253.24|valid loss: 146.71|valid f1: 81.186|valid precision: 78.458|valid recall: 84.110|valid accuracy: 81.600|valid tp: 397|valid fp: 109|valid fn: 75|valid tn: 419

 - add Tweet normalizer
2020.07.03 20:04:02|epoch:  20|train loss: 319.42|valid loss: 140.92|valid f1: 81.307|valid precision: 78.501|valid recall: 84.322|valid accuracy: 81.700|valid tp: 398|valid fp: 109|valid fn: 74|valid tn: 419

 - with multi head
2020.07.02 00:32:48|epoch:  12|train loss: 523.66|valid loss: 123.62|valid f1: 79.959|valid precision: 77.273|valid recall: 82.839|valid accuracy: 80.400|valid tp: 391|valid fp: 115|valid fn: 81|valid tn: 413
'''

