from typing import Dict, Any
import keras.utils as image
import torch
from torch.utils.data import Dataset
import os
import numpy as np

class WasteDataset(Dataset):
    def __init__(self,
                 obj_paths: Dict[str, list],
                 transform: Any=None,
                 target_transform: Any=None,
                 augment_functions: list=[]) -> None:
        """"""
        self.labels = list(obj_paths.keys())

        self.img_labels = []
        self.img_paths = []
        self.preprocess = []

        print('Number of labels: ', len(self.labels))
        for i in range(len(self.labels)):
            for path in obj_paths[self.labels[i]]:
                self.img_paths.append(path)
                self.img_labels.append(i)
                self.preprocess.append(None)
                for function in augment_functions:
                    self.img_paths.append(path)
                    self.img_labels.append(i)
                    self.preprocess.append(function)

        print('Number of samples: ', len(self.img_paths))

        self.transform = transform
        self.target_transform = target_transform


    def __len__(self) -> int:
        """"""
        return len(self.img_paths)


    def __getitem__(self,
                    idx: int) -> set([torch.Tensor, int]):
        """"""
        img_path = self.img_paths[idx]
        img_array = image.load_img(os.path.join(img_path), target_size = (224,224))
        # img_array = image.img_to_array(img)

        label = self.img_labels[idx]
        if self.transform:
            img_array = self.transform(img_array)
        else:
            img_array = torch.Tensor(image.img_to_array(img_array)).view((3, 224, 224))

        if self.target_transform:
            label = self.target_transform(label)

        if self.preprocess[idx] is not None:
            img_array = self.preprocess[idx](img_array)

        return img_array, label


    def set_transform(self,
                      transforms: None) -> None:
        """"""
        self.transform = transforms


    def n_classes(self) -> int:
        """"""
        return len(self.labels)


    def get_all_classes(self) -> list:
        """"""
        return self.labels.copy()
    
    
    def get_non_empty_classes(self,
                              return_index: bool=False) -> list:
        """"""
        if return_index:
            return np.unique(self.img_labels)
        
        return np.array(self.labels)[np.unique(self.img_labels)]