import os

import torch
from torchvision.datasets import ImageFolder, DatasetFolder
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import Dataset

from PIL import Image
import numpy as np


class iCoSegDataset(Dataset):
    def __init__(self, images_path, masks_path, image_size=224):
        self.images_path = images_path
        self.masks_path = masks_path
        self.image_size = (image_size, image_size)

        preprocess = Compose([
            ToTensor()
        ])

        self.images = DatasetFolder(
            root=images_path,
            loader=self.iCoSegImageLoader,
            extensions=("jpg"),
            transform=preprocess
        )
        self.masks = DatasetFolder(
            root=masks_path,
            loader=self.iCoSegMaskLoader,
            extensions=("png"),
        )

        self.length = len(self.images)

    def iCoSegImageLoader(self, path):
        image = Image.open(path)
        image = image.resize(self.image_size)
        #image = np.array(image, dtype=np.float32)

        return image

    def iCoSegMaskLoader(self, path):
        mask = Image.open(path)
        mask = mask.resize(self.image_size)
        mask = np.array(mask, dtype=np.uint8)

        return mask
    
    def __getitem__(self, index):

        if type(index) == torch.Tensor:
            index = index.item()
        
        image, image_label = self.images[index]
        mask, mask_label = self.masks[index]

        sample = {
            "image": image,
            "mask": mask,
            "label": image_label
        }

        return sample
    
    def __len__(self):
        return self.length

    


