import copy

import torch

from Factory import Factory
from BiGruSelfattention import BiGruSelfattention
from BiGruSelfattentionWithCheating import BiGruSelfattentionWithCheating
from Cnn import Cnn


class ModelFactory:
    def __init__(self, device='cpu', hyper_params={}):
        self.initial_train_batcher, self.initial_valid_batcher, self.criterion = Factory().get_instance()

        self.hyper_params = self.init_hyperparameters()
        self.hyper_params = {key: hyper_params[key] if key in hyper_params.keys() else val for key, val in self.hyper_params.items()}

        self.train_batcher = copy.deepcopy(self.initial_train_batcher.set_batch_size(self.hyper_params['train_batch_size']))
        self.valid_batcher = copy.deepcopy(self.initial_valid_batcher)

        self.device = torch.device(device)
        if self.hyper_params['model'] == 'gru_with_cheating':
            self.model = BiGruSelfattentionWithCheating(device=self.device, hyper_params=self.hyper_params)
        elif self.hyper_params['model'] == 'gru':
            self.model = BiGruSelfattention(device=self.device, hyper_params=self.hyper_params)
        elif self.hyper_params['model'] == 'cnn':
            self.model = Cnn(device=self.device, hyper_params=self.hyper_params)
        else:
            self.model = None
        print(self.model)

        for parameter in self.model.parameters():
            if not parameter.requires_grad:
                parameter.requires_grad = True
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
                        'embeddings': ['ntua'],
                        'model': 'gru'
                        }
        if hyper_params['optimizer'] == 'sgd':
            hyper_params['momentum'] = 0.0

        if hyper_params['model'] == 'cnn':
            hyper_params['kernel_size'] = 100
            hyper_params['window_size1'] = 3
            hyper_params['window_size2'] = 5
            hyper_params['window_size3'] = 7

        return hyper_params

    def generate(self):
        return {'model': self.model,
                'batchers': {'train': self.train_batcher, 'valid': self.valid_batcher},
                'optimizer': self.optimizer,
                'criterion': self.criterion
                }