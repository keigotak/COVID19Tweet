import copy

import torch
import torch.nn as nn

import optuna
from optuna.samplers import TPESampler

from RawEmbedding import RawEmbedding
from NtuaTwitterEmbedding import NtuaTwitterEmbedding
from StanfordTwitterEmbedding import StanfordTwitterEmbedding
from AbsolutePositionalEmbedding import AbsolutePositionalEmbedding
from Attention import Attention

from Batcher import Batcher
from DataPooler import DataPooler
from ValueWatcher import ValueWatcher
from Metrics import get_metrics, get_print_keys
from HelperFunctions import get_now, get_milliseconds, get_datasets, get_stop_words, get_tokenizer

stop_words = get_stop_words()
tokenizer = get_tokenizer()


class BiGruSelfattention(nn.Module):
    def __init__(self, device='cpu', hyper_params=None):
        super(BiGruSelfattention, self).__init__()
        self.embeddings = nn.ModuleList([self.__get_embeddings(key=key, device=device, stop_words=stop_words, tokenizer=tokenizer) for key in self.__get_embedding_keys()])

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

        self.device = device
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
        avg_seq_logits /= self.num_head

        pooled_logits = self.pooling(avg_seq_logits.transpose(2, 1)).transpose(2, 1).squeeze()
        output = self.output(pooled_logits)
        return output

    @staticmethod
    def __get_embedding_keys():
        # return ['ntua', 'stanford', 'raw', 'position']
        # return ['position']
        # return ['ntua']
        return ['raw']

    @staticmethod
    def __get_embeddings(key, device, stop_words, tokenizer):
        if key == 'ntua':
            return NtuaTwitterEmbedding(device=device, stop_words=stop_words, tokenizer=tokenizer)
        elif key == 'stanford':
            return StanfordTwitterEmbedding(device=device, stop_words=stop_words, tokenizer=tokenizer)
        elif key == 'raw':
            return RawEmbedding(device=device, stop_words=stop_words, tokenizer=tokenizer)
        elif key == 'position':
            return AbsolutePositionalEmbedding(device=device)


class BaseFactory:
    def __init__(self):
        datasets, tags = get_datasets()
        self.initial_train_batcher = Batcher(x=[pairs[0] for pairs in datasets['train']],
                                             y=[pairs[1] for pairs in datasets['train']])
        self.initial_valid_batcher = Batcher(x=[pairs[0] for pairs in datasets['valid']],
                                             y=[pairs[1] for pairs in datasets['valid']])
        self.criterion = nn.BCEWithLogitsLoss()


class Factory(BaseFactory):
    def __init__(self, device='cpu', hyper_params={}):
        super(Factory, self).__init__()
        self.hyper_params = self.init_hyperparameters()
        self.hyper_params = {key: hyper_params[key] if key in hyper_params.keys() else val for key, val in self.hyper_params.items()}

        self.train_batcher = copy.deepcopy(self.initial_train_batcher.set_batch_size(self.hyper_params['train_batch_size']))
        self.valid_batcher = copy.deepcopy(self.initial_valid_batcher)

        self.device = torch.device(device)
        self.model = BiGruSelfattention(device=self.device, hyper_params=self.hyper_params)
        print(self.model)

        for parameter in self.model.parameters():
            if not parameter.requires_grad:
                print(parameter)

        if self.hyper_params['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hyper_params['lr'], momentum=self.hyper_params['momentum'])
        elif hyper_params['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyper_params['lr'])

    @staticmethod
    def init_hyperparameters():
        hyper_params = {'lr': 0.01,
                        'num_head': 8,
                        'dropout_ratio': 0.2,
                        'train_batch_size': 4,
                        'weight_decay': 0.0,
                        'clip_grad_nurm': 0.0,
                        'optimizer': 'sgd',
                        'momentum': 0.0
                        }
        return hyper_params

    def generate(self):
        return {'model': self.model,
                'batchers': {'train': self.train_batcher, 'valid': self.valid_batcher},
                'optimizer': self.optimizer,
                'criterion': self.criterion
                }


class Runner:
    def __init__(self, device='cuda:0', hyper_params={}):
        self.device = device
        factory = Factory(device=self.device, hyper_params=hyper_params).generate()
        self.model, self.batchers, self.optimizer, self.criterion = [factory[key] for key in ['model', 'batchers', 'optimizer', 'criterion']]
        self.TRAIN_MODE, self.VALID_MODE = 'train', 'valid'
        self.batchers = {self.TRAIN_MODE: self.batchers[self.TRAIN_MODE], self.VALID_MODE: self.batchers[self.VALID_MODE]}
        self.poolers = {self.VALID_MODE: DataPooler()}
        self.valuewatcher = ValueWatcher()
        self.epochs = 100

    def run(self):
        for e in range(self.epochs):
            running_loss = {key: 0.0 for key in [self.TRAIN_MODE, self.VALID_MODE]}

            mode = self.TRAIN_MODE
            self.model.train()
            while not self.batchers[mode].is_batch_end():
                self.optimizer.zero_grad()
                x_batch, y_batch = self.batchers[mode].get_batch()
                outputs = self.model(x_batch)
                labels = torch.Tensor(y_batch).float().unsqueeze(1).to(self.device)
                loss = self.criterion(outputs, labels)
                loss.backward()
                running_loss[mode] += loss.item()
                self.optimizer.step()
            self.batchers[mode].reset(with_shuffle=True)

            mode = self.VALID_MODE
            self.model.eval()
            with torch.no_grad():
                while not self.batchers[mode].is_batch_end():
                    x_batch, y_batch = self.batchers[mode].get_batch()
                    outputs = self.model(x_batch)
                    labels = torch.Tensor(y_batch).float().unsqueeze(1).to(self.device)
                    loss = self.criterion(outputs, labels)
                    running_loss[mode] += loss.item()

                    preds = torch.sigmoid(outputs)
                    predicted_label = [0 if item < 0.5 else 1 for item in preds.squeeze().tolist()]
                    self.poolers[mode].set('epoch{}-x'.format(e + 1), x_batch)
                    self.poolers[mode].set('epoch{}-y'.format(e + 1), y_batch)
                    self.poolers[mode].set('epoch{}-logits'.format(e + 1), outputs.squeeze().tolist())
                    self.poolers[mode].set('epoch{}-preds'.format(e + 1), preds.squeeze().tolist())
                    self.poolers[mode].set('epoch{}-predicted_label'.format(e + 1), predicted_label)

            metrics = get_metrics(self.poolers[mode].get('epoch{}-predicted_label'.format(e + 1)),
                                  self.poolers[mode].get('epoch{}-y'.format(e + 1)))
            self.poolers[mode].set('epoch{}-metrics'.format(e + 1), metrics)
            self.poolers[mode].set('epoch{}-train_loss'.format(e + 1), running_loss[self.TRAIN_MODE])
            self.poolers[mode].set('epoch{}-valid_loss'.format(e + 1), running_loss[self.VALID_MODE])
            self.valuewatcher.update(metrics['f1'])
            self.batchers[mode].reset(with_shuffle=False)

            now = get_now()
            text_line = '|'.join(['{} {}: {:.3f}'.format(mode, key, 100 * metrics[key])
                                  if key not in set(['tp', 'fp', 'fn', 'tn'])
                                  else '{} {}: {}'.format(mode, key, metrics[key])
                                  for key in get_print_keys()])
            print('{}|epoch: {:3d}|train loss: {:.2f}|valid loss: {:.2f}|{}'.format(now,
                                                                                    e + 1,
                                                                                    running_loss[self.TRAIN_MODE],
                                                                                    running_loss[self.VALID_MODE],
                                                                                    text_line))

            if self.valuewatcher.is_over():
                break
        return metrics['f1']

    def export_results(self):
        pass




class HyperparameterSearcher:
    def __init__(self):
        self.direction = 'maximize'
        self.sampler = TPESampler(seed=int(get_milliseconds()))
        self.study_name = 'test'
        # study = optuna.create_study(study_name=study_name, sampler=sampler, direction=direction, storage='sqlite:///./results/hyp-search.db', load_if_exists=True)
        self.study = optuna.create_study(study_name=self.study_name, sampler=self.sampler, direction=self.direction)

    def get_hyperparameters(self, trial):
        hyper_params = {}
        hyper_params['optimizer_type'] = trial.suggest_categorical('optimizer', ['sgd', 'adam'])
        hyper_params['learning_ratio'] = trial.suggest_loguniform('learning_rate', 1e-5, 5e-1)
        hyper_params['gradient_clip'] = trial.suggest_uniform('gradient_clip', 0.0, 5.0)
        hyper_params['weight_decay'] = trial.suggest_uniform('weight_decay', 0.0, 1.0)
        hyper_params['dropout_ratio'] = trial.suggest_uniform('dropout_ratio', 0.0, 1.0)
        hyper_params['num_head'] = trial.suggest_int('num_head', 1, 16, 1)
        return hyper_params

    def objective(self, trial):
        hyper_params = self.get_hyperparameters(trial=trial)
        runner = Runner(hyper_params=hyper_params, device='cuda:0')
        score = runner.run()
        return score

    def run(self):
        self.study.optimize(self.objective,
                            n_trials=10000,
                            catch=(ValueError,),
                            n_jobs=4)
        print(self.study.best_params)



if __name__ == '__main__':
    runner = Runner()
    score = runner.run()

    # hyps = HyperparameterSearcher()
    # hyps.run()



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

