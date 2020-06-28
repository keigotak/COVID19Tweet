import torch
import torch.nn as nn

from RawEmbedding import RawEmbedding
from Attention import Attention


class BiGruSelfattention(nn.Module):
    def __init(self):
        super(BiGruSelfattention, self).__init__()
        self.embedding = RawEmbedding()
        emb_dim = self.embedding.embedding_dim
        self.f_rnn1 = nn.GRU(input_size=emb_dim, hidden_dim=emb_dim, batch_first=True)
        self.b_rnn1 = nn.GRU(input_size=emb_dim, hidden_dim=emb_dim, batch_first=True)
        self.f_rnn2 = nn.GRU(input_size=emb_dim, hidden_dim=emb_dim, batch_first=True)
        self.b_rnn2 = nn.GRU(input_size=emb_dim, hidden_dim=emb_dim, batch_first=True)
        self.attention = Attention(dimentions=emb_dim)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)

    def forward(self, batch_sentence):
        embeddings = self.embedding(batch_sentence)
        max_len = max(map(len, batch_sentence))

        outf1, hidf1 = self.f_gru1(self.dropout1(embeddings))
        resf1 = outf1 + embeddings
        rev_resf1 = resf1[:,torch.arange(max_len-1, -1, -1),:] # reversed

        outb1, hidb1 = self.b_gru1(self.dropout1(rev_resf1))
        resb1 = outb1 + rev_resf1
        rev_resb1 = resb1[:,torch.arange(max_len-1, -1, -1),:] # not reversed

        outf2, hidf2 = self.f_gru2(self.dropout2(rev_resb1))
        resf2 = outf2 + rev_resb1
        rev_resf2 = resf2[:,torch.arange(max_len-1, -1, -1),:] # reversed

        outb2, hidb2 = self.b_gru2(self.dropout2(rev_resf2))
        resb2 = outb2 + rev_resf2
        rev_resb2 = resb2[:,torch.arange(max_len-1, -1, -1),:] # not reversed

        outf3, hidf3 = self.f_gru3(self.dropout3(rev_resb2))
        resf3 = outf3 + rev_resb2
        rev_resf3 = resf3[:,torch.arange(max_len-1, -1, -1),:] # reversed

        outb3, hidb3 = self.b_gru3(self.dropout3(rev_resf3))
        resb3 = outb3 + rev_resf3
        rev_resb3 = resb3[:,torch.arange(max_len-1, -1, -1),:] # not reversed

        drop_output = self.dropout4(rev_resb3)
        # flat_output = torch.flatten(drop_output, start_dim=1)
        classified_logits, seq_logits, attention_weights = self.attention(query=drop_output, context=drop_output)

        return classified_logits


if __name__ == '__main__':
    from HelperFunctions import get_datasets
    datasets, tags = get_datasets()
    sentences = [pairs[0] for pairs in datasets['train']]
    model = BiGruSelfattention()
    model(sentences)

