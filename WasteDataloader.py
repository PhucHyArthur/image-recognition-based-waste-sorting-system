from torch.utils.data import DataLoader, Dataset
from WasteDataset import WasteDataset

def generate_batches(dataset: Dataset,
                     batch_size: int,
                     shuffle: bool=True,
                     drop_last: bool=True) -> DataLoader:
    """"""
    return DataLoader(dataset=dataset, batch_size=batch_size,
                      shuffle=shuffle, drop_last=drop_last)


class WasteDataLoader():
    def __init__(self,
                 datasets: dict[str, Dataset],
                 batch_size: int=64,
                 shuffle: bool=True,
                 drop_last: bool=True) -> None:
        """"""
        self.datasets = datasets
        self.data_loader = {}
        self.sizes = {}
        self.labels = []
        self.non_empty_labels = []
        for key in datasets.keys():
            self.data_loader[key] = generate_batches(dataset=datasets[key],
                                                     batch_size=batch_size,
                                                     shuffle=shuffle,
                                                     drop_last=drop_last)
            self.sizes[key] = len(datasets[key])
            self.labels = datasets[key].get_all_classes()
            # self.non_empty_labels = datasets[key].get_non_empty_classes()


    def get_dataset(self,
                    name_of_set: str) -> WasteDataset:
        """"""
        if name_of_set in self.datasets:
            return self.datasets[name_of_set]

        return None
    

    def get_dataloader(self,
                       name_of_set: str) -> DataLoader:
        """"""
        if name_of_set in self.data_loader:
            return self.data_loader[name_of_set]

        return None


    def size_of_set(self,
                    name_of_set: str) -> int:
        """"""
        if name_of_set in self.data_loader:
            return self.sizes[name_of_set]

        return 0


    def n_classes(self) -> int:
        """"""
        return len(self.labels)


    def get_all_classes(self) -> list:
        """"""
        return self.labels.copy()
    
    
    def get_non_empty_classes(self) -> list:
        """"""
        return self.non_empty_labels.copy()