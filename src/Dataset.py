from HelperFunctions import get_datasets, get_detailed_datasets


class Dataset:
    __singleton = None
    __datasets = None
    __modes = None

    def __new__(cls):
        if cls.__singleton is None:
            cls.__singleton = super(Dataset, cls).__new__(cls)
            cls.__datasets, cls.__modes = get_datasets()
        return cls.__singleton

    def get_instance(self):
        return self.__datasets


class DetailedDataset:
    __singleton = None
    __datasets = None
    __modes = None

    def __new__(cls, label='created_at'):
        if cls.__singleton is None:
            cls.__singleton = super(DetailedDataset, cls).__new__(cls)
            cls.__datasets, cls.__modes = get_detailed_datasets(label=label)
        return cls.__singleton

    def get_instance(self):
        return self.__datasets