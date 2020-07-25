import torch.nn as nn

from Batcher import Batcher
from Dataset import Dataset


class Factory:
    __singleton = None
    __initial_train_batcher = None
    __initial_valid_batcher = None
    __criterion = None

    def __new__(cls):
        if cls.__singleton is None:
            cls.__singleton = super(Factory, cls).__new__(cls)
            datasets = Dataset().get_instance()
            cls.__initial_train_batcher = Batcher(x=[pairs[0] for pairs in datasets['train']],
                                                  y=[pairs[1] for pairs in datasets['train']])
            cls.__initial_valid_batcher = Batcher(x=[pairs[0] for pairs in datasets['valid']],
                                                  y=[pairs[1] for pairs in datasets['valid']])
            cls.__criterion = nn.BCEWithLogitsLoss()
        return cls.__singleton

    def get_instance(self):
        return self.__initial_train_batcher, self.__initial_valid_batcher, self.__criterion


class AbstractFactory:
    def __init__(self):
        self.initial_train_batcher, self.initial_valid_batcher, self.criterion = Factory().get_instance()

    @staticmethod
    def init_hyperparameters():
        hyper_params = {'lr': 0.01,
                        'num_head': 8,
                        'dropout_ratio': 0.2,
                        'train_batch_size': 4,
                        'weight_decay': 0.0,
                        'clip_grad_nurm': 0.0,
                        'optimizer': 'sgd',
                        'momentum': 0.0,
                        'embeddings': ['ntua'],
                        'model': 'simple'
                        }
        return hyper_params

    def generate(self):
        return {'model': self.model,
                'batchers': {'train': self.train_batcher, 'valid': self.valid_batcher},
                'optimizer': self.optimizer,
                'criterion': self.criterion
                }