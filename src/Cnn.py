import torch
import torch.nn as nn

from RawEmbedding import RawEmbedding
from NtuaTwitterEmbedding import NtuaTwitterEmbedding
from StanfordTwitterEmbedding import StanfordTwitterEmbedding
from AbsolutePositionalEmbedding import AbsolutePositionalEmbedding
from Attention import Attention

from Batcher import Batcher
from DataPooler import DataPooler
from ValueWatcher import ValueWatcher
from Metrics import get_metrics, get_print_keys
from HelperFunctions import get_now


class Cnn(nn.Module):
    def __init__(self, device='cpu', hyper_params=None):
        super(Cnn, self).__init__()
        self.embeddings = nn.ModuleList([self.__get_embeddings(key=key, device=device) for key in self.__get_embedding_keys()])

        emb_dim = sum([item.embedding_dim for item in self.embeddings])
        self.hidden_size = emb_dim
        self.kernel_size = 100
        self.window_sizes = [3, 5, 7]
        self.input_chs = [1, 1, 1]
        self.output_chs = [self.kernel_size, self.kernel_size, self.kernel_size]

        self.dropout = nn.Dropout(0.5)
        self.cnns = nn.ModuleList([nn.Conv2d(in_channels=ich, out_channels=och, kernel_size=(ws, emb_dim), stride=1) for ich, och, ws in zip(self.input_chs, self.output_chs, self.window_sizes)])

        self.num_head = 8
        self.attention = nn.ModuleList([Attention(dimensions=self.kernel_size) for _ in range(self.num_head)])

        self.output = nn.Linear(sum(self.output_chs), 1)

        self.device = device
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

    @staticmethod
    def __get_embedding_keys():
        return ['raw', 'position']
        # return ['ntua', 'stanford', 'raw', 'position']
        # return ['position']

    @staticmethod
    def __get_embeddings(key, device):
        if key == 'ntua':
            return NtuaTwitterEmbedding(device=device)
        elif key == 'stanford':
            return StanfordTwitterEmbedding(device=device)
        elif key == 'raw':
            return RawEmbedding(device=device)
        elif key == 'position':
            return AbsolutePositionalEmbedding(device=device)


if __name__ == '__main__':
    from HelperFunctions import get_datasets
    datasets, tags = get_datasets()
    train_batcher = Batcher(x=[pairs[0] for pairs in datasets['train']], y=[pairs[1] for pairs in datasets['train']])
    valid_batcher = Batcher(x=[pairs[0] for pairs in datasets['valid']], y=[pairs[1] for pairs in datasets['valid']])

    device = torch.device('cuda:0')
    model = Cnn(device=device)
    print(model)

    for parameter in model.parameters():
        if not parameter.requires_grad:
            print(parameter)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
                predicted_label = [0 if item < 0.5 else 1 for item in preds.squeeze().tolist()]
                poolers[mode].set('epoch{}-x'.format(e+1), x_batch)
                poolers[mode].set('epoch{}-y'.format(e+1), y_batch)
                poolers[mode].set('epoch{}-logits'.format(e+1), outputs.squeeze().tolist())
                poolers[mode].set('epoch{}-preds'.format(e+1), preds.squeeze().tolist())
                poolers[mode].set('epoch{}-predicted_label'.format(e+1), predicted_label)

        metrics = get_metrics(poolers[mode].get('epoch{}-predicted_label'.format(e+1)), poolers[mode].get('epoch{}-y'.format(e+1)))
        poolers[mode].set('epoch{}-metrics'.format(e+1), metrics)
        poolers[mode].set('epoch{}-train_loss'.format(e+1), running_loss[TRAIN_MODE])
        poolers[mode].set('epoch{}-valid_loss'.format(e+1), running_loss[VALID_MODE])
        valuewatcher.update(metrics['f1'])
        batchers[mode].reset(with_shuffle=False)

        now = get_now()
        text_line = '|'.join(['{} {}: {:.3f}'.format(mode, key, 100 * metrics[key]) if key not in set(['tp', 'fp', 'fn', 'tn']) else '{} {}: {}'.format(mode, key, metrics[key]) for key in get_print_keys()])
        print('{}|epoch: {:3d}|train loss: {:.2f}|valid loss: {:.2f}|{}'.format(now, e+1, running_loss[TRAIN_MODE], running_loss[VALID_MODE], text_line))

        if valuewatcher.is_over():
            break


'''
'''

