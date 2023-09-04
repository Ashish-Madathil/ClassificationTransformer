from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pickle
from pdb import set_trace as stop
import json, string, sys
from dataloaders.data_utils import get_unk_mask_indices
from dataloaders.data_utils import image_loader, pil_loader
from sklearn.preprocessing import LabelEncoder


class BlastocystDataset(Dataset):
    def __init__(self, data_dir, csv_file, label_column, split='train', transform=None, known_labels=0, testing=False):
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.label_column = label_column  # Specify the label column to load ('EXP', 'ICM', or 'TE')
        self.split = split
        self.transform = transform
        self.known_labels = known_labels
        self.testing = testing

        self.data = pd.read_csv(csv_file)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        img_filename = row['Image']
        img_path = os.path.join(self.data_dir, img_filename)

        image = image_loader(img_path, self.transform)
        # labels = self.data[self.label_column].unique().tolist()
        labels = row[self.label_column]
        # mask = self.prepare_mask(label)

        # labels = torch.Tensor(labels)
        # labels = torch.Tensor([labels])
        # mask = labels.clone()

        
        self.num_labels = len(self.data[self.label_column].unique().tolist())

        labels = torch.Tensor([labels] * self.num_labels)
        mask = labels.clone()

        unk_mask_indices = get_unk_mask_indices(image, self.testing, self.num_labels, self.known_labels)
        mask.scatter_(0, torch.Tensor(unk_mask_indices).long(), -1)

        sample = {
            'image': image,
            'labels': labels,
            'mask': mask,
            'imageIDs': str(img_filename)
        }

        return sample
    
    def prepare_mask(self, label):
        mask = torch.full((self.num_labels,), -1, dtype=torch.float32)  # Initialize mask with -1
        mask[label] = 1  # Set the mask value for the true label to 1
        return mask
