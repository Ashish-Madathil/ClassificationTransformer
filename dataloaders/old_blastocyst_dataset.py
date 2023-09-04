
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
    def __init__(self, data_dir, csv_file, split='train', transform=None, known_labels=0, testing=False):
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.split = split
        self.transform = transform
        self.known_labels = known_labels
        self.testing = testing

        self.data = pd.read_csv(csv_file)

        self.label_encoders = {
            'EXP': LabelEncoder(),
            'ICM': LabelEncoder(),
            'TE': LabelEncoder()
        }

        for col in self.label_encoders:
            self.label_encoders[col].fit(self.data[col])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        img_filename = row['Image']
        img_path = os.path.join(self.data_dir, img_filename)

        image = image_loader(img_path, self.transform)

        labels = torch.tensor([
            self.label_encoders['EXP'].transform([str(row['EXP'])]),
            self.label_encoders['ICM'].transform([str(row['ICM'])]),
            self.label_encoders['TE'].transform([str(row['TE'])])
        ], dtype=torch.long).squeeze()

        mask = labels.clone()
        self.num_labels = len(labels)
        unk_mask_indices = get_unk_mask_indices(image, self.testing,self.num_labels,self.known_labels)
        mask.scatter_(0, torch.Tensor(unk_mask_indices).long(), -1)

        sample = {
            'image': image,
            'labels': labels,
            'mask': mask,
            'imageIDs': str(img_filename)
        }

        return sample
        
   

