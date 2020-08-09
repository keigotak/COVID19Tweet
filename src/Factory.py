import torch.nn as nn

from Batcher import Batcher
from Dataset import Dataset, DetailedDataset
from HelperFunctions import get_modes


class Factory:
    __singleton = None
    __initial_train_batcher = None
    __initial_valid_batcher = None
    __criterion = None

    def __new__(cls, label='informative_or_not'):
        if cls.__singleton is None:
            cls.__singleton = super(Factory, cls).__new__(cls)

            if label in {'informative_or_not'}:
                datasets = Dataset().get_instance()
                cls.__criterion = nn.BCEWithLogitsLoss()
            else:
                datasets = DetailedDataset(label=label).get_instance()
                labels = set()
                for mode in get_modes():
                    for _, label in datasets[mode]:
                        labels.add(label)
                labels = {key: i for i, key in enumerate(sorted(labels))}

                for mode in get_modes():
                    for i, (_, label) in enumerate(datasets[mode]):
                        datasets[mode][i][1] = labels[label]

                cls.__criterion = nn.CrossEntropyLoss()

            cls.__initial_train_batcher = Batcher(x=[pairs[0] for pairs in datasets['train']],
                                                  y=[pairs[1] for pairs in datasets['train']])
            cls.__initial_valid_batcher = Batcher(x=[pairs[0] for pairs in datasets['valid']],
                                                  y=[pairs[1] for pairs in datasets['valid']])

        return cls.__singleton

    def get_instance(self):
        return self.__initial_train_batcher, self.__initial_valid_batcher, self.__criterion
