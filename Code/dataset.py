
###################################
########## import library #########
###################################

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from sklearn import metrics

from skimage import io, color

import time
import os
import pickle

import matplotlib.pyplot as plt
import scikitplot as skplt


###################################
########### dataloader ############
###################################

class ChestXray(Dataset):

    def __init__(self, csv_file, image_root_dir, transform=None):

        self.data_frame = pd.read_csv(csv_file)
        self.image_root_dir = image_root_dir
        self.image_path = self.data_frame['image_index']
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        image_filename = self.image_root_dir + self.image_path[index]
        image = io.imread(image_filename, as_gray=True)

        sample = {}

        # need to transpose: input size for ToPILImage() is H*W*C not C*H*W
        image = np.repeat(image[None,...], 3, axis=0).transpose(1, 2, 0)

        if self.transform:
            image = self.transform(np.uint8(image))

        sample['image'] = image

        label_col_names = ['normal', 'pneumonia']

        sample['label'] = torch.LongTensor([self.data_frame['label'][index]])

        return sample
