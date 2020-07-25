from pathlib import Path

import torch
import torch.nn as nn

from AbstractModel import AbstractModel
from Attention import Attention
from StopWords import StopWords


class BiGruSelfattentionWithCheating(AbstractModel):
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
        self.output = nn.Linear(emb_dim + 1, 1)

        self.to(device)

        with Path('../data/utils/cheatsheet.txt').open('r', encoding='utf-8-sig') as f:
            self.cheatsheet = set([line.strip() for line in f.readlines()])

        self.added_stop_words = StopWords(with_applied=True).get_instance()

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

        cheat_scores = self.__cheating_output(batch_sentence=batch_sentence)
        output = self.output(torch.cat((pooled_logits, torch.tensor(cheat_scores).unsqueeze(1).float().to(self.device)), dim=1))
        return output

    def __cheating_output(self, batch_sentence):
        cheating_scores = []

        for sentence in batch_sentence:
            words = tokenizer.pre_process_doc(sentence)
            del_words = [word for word in words if word not in self.added_stop_words]
            tri_grams = ['_'.join([del_words[i], del_words[i+1], del_words[i+2]]) for i in range(len(del_words[:-2]))]

            if len(set(tri_grams) & self.cheatsheet) >= 1:
                cheating_scores.append(1)
            else:
                cheating_scores.append(0)

        return cheating_scores


'''
'''

