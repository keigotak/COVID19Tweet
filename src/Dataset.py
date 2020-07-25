from HelperFunctions import get_datasets


class Dataset:
    __singleton = None
    __datasets = None
    __tags = None

    def __new__(cls):
        if cls.__singleton is None:
            cls.__singleton = super(Dataset, cls).__new__(cls)
            cls.__datasets, cls.__tags = get_datasets()
        return cls.__singleton

    def get_instance(self):
        return self.__datasets
