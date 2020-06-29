import torch
import torch.nn as nn

from RawEmbedding import RawEmbedding
from Attention import Attention

from Batcher import Batcher


class BiGruSelfattention(nn.Module):
    def __init__(self):
        super(BiGruSelfattention, self).__init__()
        self.embedding = RawEmbedding()
        emb_dim = self.embedding.embedding_dim
        self.f_gru1 = nn.GRU(input_size=emb_dim, hidden_size=emb_dim, batch_first=True)
        self.b_gru1 = nn.GRU(input_size=emb_dim, hidden_size=emb_dim, batch_first=True)
        self.f_gru2 = nn.GRU(input_size=emb_dim, hidden_size=emb_dim, batch_first=True)
        self.b_gru2 = nn.GRU(input_size=emb_dim, hidden_size=emb_dim, batch_first=True)
        self.attention = Attention(dimensions=emb_dim)
        self.hidden_size = emb_dim

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Linear(emb_dim, 1)

    def forward(self, batch_sentence):
        embeddings = self.embedding(batch_sentence)
        max_len = embeddings.shape[1]

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

        drop_output = self.dropout3(rev_resb2)
        seq_logits, attention_weights = self.attention(query=drop_output, context=drop_output)

        pooled_logits = self.pooling(seq_logits.transpose(2, 1)).transpose(2, 1).squeeze()
        output = self.output(pooled_logits)
        return output


if __name__ == '__main__':
    from HelperFunctions import get_datasets
    datasets, tags = get_datasets()
    train_batcher = Batcher(x=[pairs[0] for pairs in datasets['train']], y=[pairs[1] for pairs in datasets['train']])
    valid_batcher = Batcher(x=[pairs[0] for pairs in datasets['valid']], y=[pairs[1] for pairs in datasets['valid']])

    model = BiGruSelfattention()
    print(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    model.train()
    epochs = 100
    for e in range(epochs):
        running_loss = 0.0
        while not train_batcher.is_batch_end():
            optimizer.zero_grad()
            x_batch, y_batch = train_batcher.get_batch()
            outputs = model(x_batch)
            labels = torch.Tensor(y_batch).float().unsqueeze(1)
            loss = criterion(outputs, labels)
            optimizer.step()
            running_loss += loss.item()
        train_batcher.reset(with_shuffle=True)
        print('loss: {}'.format(running_loss))
