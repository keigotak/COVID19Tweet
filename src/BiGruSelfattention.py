import torch
import torch.nn as nn

from RawEmbedding import RawEmbedding
from Attention import Attention

from Batcher import Batcher
from DataPooler import DataPooler
from ValueWatcher import ValueWatcher
from Metrics import get_metrics, get_print_keys
from HelperFunctions import get_now


class BiGruSelfattention(nn.Module):
    def __init__(self, device='cpu'):
        super(BiGruSelfattention, self).__init__()
        self.embedding = RawEmbedding(device=device)
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

        self.device = device
        self.to(device)

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

    device = torch.device('cuda:0')
    model = BiGruSelfattention(device=device)
    print(model)

    for parameter in model.parameters():
        if not parameter.requires_grad:
            print(parameter)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

    TRAIN_MODE, VALID_MODE = 'train', 'valid'
    batchers = {TRAIN_MODE: train_batcher, VALID_MODE: valid_batcher}
    poolers = {VALID_MODE: DataPooler()}
    valuewatcher = ValueWatcher()
    epochs = 100

    for e in range(epochs):
        running_loss = {key: 0.0 for key in [TRAIN_MODE, VALID_MODE]}

        mode = TRAIN_MODE
        model.train()
        while not batchers[mode].is_batch_end():
            optimizer.zero_grad()
            x_batch, y_batch = batchers[mode].get_batch()
            outputs = model(x_batch)
            labels = torch.Tensor(y_batch).float().unsqueeze(1).to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            running_loss[mode] += loss.item()
            optimizer.step()
        batchers[mode].reset(with_shuffle=True)

        mode = VALID_MODE
        model.eval()
        with torch.no_grad():
            while not batchers[mode].is_batch_end():
                x_batch, y_batch = batchers[mode].get_batch()
                outputs = model(x_batch)
                labels = torch.Tensor(y_batch).float().unsqueeze(1).to(device)
                loss = criterion(outputs, labels)
                running_loss[mode] += loss.item()

                preds = torch.sigmoid(outputs)
                predict_label = [0 if item < 0.5 else 1 for item in preds.squeeze().tolist()]
                poolers[mode].set('epoch{}-x'.format(e+1), x_batch)
                poolers[mode].set('epoch{}-y'.format(e+1), y_batch)
                poolers[mode].set('epoch{}-logits'.format(e+1), outputs.squeeze().tolist())
                poolers[mode].set('epoch{}-preds'.format(e+1), preds.squeeze().tolist())
                poolers[mode].set('epoch{}-predict_label'.format(e+1), predict_label)

        metrics = get_metrics(poolers[mode].get('epoch{}-predict_label'.format(e+1)), poolers[mode].get('epoch{}-y'.format(e+1)))
        poolers[mode].set('epoch{}-metrics'.format(e+1), metrics)
        poolers[mode].set('epoch{}-train_loss'.format(e+1), running_loss[TRAIN_MODE])
        poolers[mode].set('epoch{}-valid_loss'.format(e+1), running_loss[VALID_MODE])
        valuewatcher.update(metrics['f1'])
        batchers[mode].reset(with_shuffle=False)

        now = get_now()
        text_line = '|'.join(['{} {}: {:.3f}'.format(mode, key, 100 * metrics[key]) if key not in set(['tp', 'fp', 'fn', 'tn']) else '{} {}: {}'.format(mode, key, metrics[key]) for key in get_print_keys() ])
        print('{}|epoch: {:3d}|train loss: {:.2f}|valid loss: {:.2f}|{}'.format(now, e+1, running_loss[TRAIN_MODE], running_loss[VALID_MODE], text_line))

        if valuewatcher.is_over():
            break


