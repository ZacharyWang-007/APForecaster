from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
import random


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB') 
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class Climate_dataloader(Dataset):
    def __init__(self, dataset, mean=None, std=None):
        self.dataset = dataset
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.dataset)
    
    def read_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = torch.Tensor(img)[None, ]
        return img

    def read_img_float32(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = torch.Tensor(img)[None, ]
        return img

    def temporal_embedding(self, date):
        _, month0, day0 = date.split('_')
        month0, day0 = int(month0)-1, int(day0)-1

        temp_m, temp_d = torch.zeros(12), torch.zeros(31)
        temp_m[month0], temp_d[day0] = 1, 1
        return torch.cat([temp_m, temp_d], dim=0)
            

    def __getitem__(self, index):
        # id0, id1, id2, id3, id4, id5, id6, id7, id8, wildfire, wildfire_target, fire_o3, fire_pm25, total_o3, total_pm25 = self.dataset[index]
        temp = self.dataset[index]

        indicators_today_files = torch.load(temp[0])
        indicators_target_files = torch.load(temp[1])
    

        # wild_fire_today, wild_fire_target = temp[2], temp[3]
        total_O3_today, total_O3_target = temp[2], temp[3]

        # wild_fire_today, wild_fire_target = self.read_img(wild_fire_today), self.read_img(wild_fire_target)

        total_O3_today = self.read_img_float32(total_O3_today)
        total_O3_target = self.read_img_float32(total_O3_target)

        today_embedding, target_embedding = self.temporal_embedding(temp[4]), self.temporal_embedding(temp[5])
        target_data = temp[6]
       
        return indicators_today_files, indicators_target_files, total_O3_today, total_O3_target, today_embedding, target_embedding, target_data




       
   