import pickle
from torch.utils.data import DataLoader, Dataset
from prefetch_generator import BackgroundGenerator
import PIL.Image as Image
import torch
import os
import numpy as np

class DataLoaderX(DataLoader):
    # speed up the loading data process
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class myDataset(Dataset):
    def __init__(self, datapath, images, coordinates, transform=None):
        self.image = images
        self.landmark = coordinates
        self.transform = transform
        self.datapath = datapath
        # self.images = [Image.open(os.path.join(self.datapath, image_path)).copy() for image_path in images]

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        img = Image.open(f'{self.datapath}/{self.image[idx]}')
        # img = self.images[idx]
        landmark = self.landmark[idx]
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(landmark).float()


def get_dataloader(train_transform, valid_transform, batch):    
    data_path =  './data'
    train_path = f'{data_path}/synthetics_train'
    valid_path = f'{data_path}/aflw_val'

    with open(f'{train_path}/annot.pkl', "rb") as f:
        train_obj = pickle.load(f)
    with open(f'{valid_path}/annot.pkl', "rb") as f:
        valid_obj = pickle.load(f)
        
    train_img, train_landmark = train_obj
    valid_img, valid_landmark = valid_obj

    train_set = myDataset(train_path, train_img, train_landmark, train_transform)
    valid_set = myDataset(valid_path, valid_img, valid_landmark, valid_transform)

    train_loader = DataLoaderX(train_set, batch_size=batch, shuffle=True)
    valid_loader = DataLoaderX(valid_set, batch_size=batch, shuffle=False)

    return train_loader, valid_loader
