import os
import warnings

import mlconfig
from torch.utils import data
from torchvision import datasets, transforms
import numpy as np
import skimage.io as io

@mlconfig.register
class DataLoader(data.DataLoader):

    def __init__(self, train: bool, batch_size: int, root="/home/cchoi/fire/efficientnet-pytorch/efficientnet/datasets/fire_data/", image_size=224, **kwargs):
        normalize = transforms.Normalize(mean=[114.73675626, 103.49747355, 93.53266762], std=[70.83235318, 67.77305118, 75.46882162]) ###

        warnings.filterwarnings('ignore', category=UserWarning)

        if train:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
            self.num = 27321
            self.lb_file = np.load("/home/cchoi/fire/efficientnet-pytorch/efficientnet/datasets/fire_data/trainlb.npy")
            self.root = root + "/train/"
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
            self.num = 1692
            self.lb_file = np.load("/home/cchoi/fire/efficientnet-pytorch/efficientnet/datasets/fire_data/testlb.npy")
            self.root = root + "/test/"
    
    def __len__(self):
        return self.num

    
    def __getitem__(self, idx):
        lb = self.lb_file[idx]
        img = self.transform(io.imread(self.root + str(idx) + ".jpg"))
        return (img, lb)
