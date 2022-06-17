import pickle
from torch.utils.data import DataLoader, Dataset
import PIL.Image as Image
import torch
import os

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

        delta = abs(landmark[36][0] - landmark[45][0])
        weight = (384.0 - delta) / 384.0

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(landmark).float(), weight

class TestDataset(Dataset):

    def __init__(self, datapath, transform):
        self.datapath = datapath
        self.transform = transform
        self.img_fname = sorted(os.listdir(datapath))

    def __len__(self):
        return len(self.img_fname)

    def __getitem__(self, idx):
        img = Image.open(f'{self.datapath}/{self.img_fname[idx]}')
        if self.transform:
            img = self.transform(img)
        return self.img_fname[idx], img


def get_dataloader(data_path, train_transform, valid_transform, batch, num_workers): 
    # data_path =  './data'
    train_path = f'{data_path}/synthetics_train'
    valid_path = f'{data_path}/aflw_val'
    test_path = f'{data_path}/aflw_test'

    with open(f'{train_path}/annot.pkl', "rb") as f:
        train_obj = pickle.load(f)
    with open(f'{valid_path}/annot.pkl', "rb") as f:
        valid_obj = pickle.load(f)
        
    train_img, train_landmark = train_obj
    valid_img, valid_landmark = valid_obj

    train_set = myDataset(train_path, train_img, train_landmark, train_transform)
    valid_set = myDataset(valid_path, valid_img, valid_landmark, valid_transform)
    test_set = TestDataset(test_path, valid_transform)

    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch, shuffle=False)

    return train_loader, valid_loader, test_loader
