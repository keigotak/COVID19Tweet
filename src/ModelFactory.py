import copy

import torch

from AbstractFactory import AbstractFactory
from BiGruSelfattention import BiGruSelfattention
from BiGruSelfattentionWithCheating import BiGruSelfattentionWithCheating


class ModelFactory(AbstractFactory):
    def __init__(self, device='cpu', hyper_params={}):
        super(ModelFactory, self).__init__()
        self.hyper_params = self.init_hyperparameters()
        self.hyper_params = {key: hyper_params[key] if key in hyper_params.keys() else val for key, val in self.hyper_params.items()}

        self.train_batcher = copy.deepcopy(self.initial_train_batcher.set_batch_size(self.hyper_params['train_batch_size']))
        self.valid_batcher = copy.deepcopy(self.initial_valid_batcher)

        self.device = torch.device(device)
        if self.hyper_params['model'] == 'gru_with_cheating':
            self.model = BiGruSelfattentionWithCheating(device=self.device, hyper_params=self.hyper_params)
        else:
            self.model = BiGruSelfattention(device=self.device, hyper_params=self.hyper_params)
        print(self.model)

        for parameter in self.model.parameters():
            if not parameter.requires_grad:
                print(parameter)

        if self.hyper_params['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hyper_params['lr'], momentum=self.hyper_params['momentum'])
        elif hyper_params['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyper_params['lr'])
