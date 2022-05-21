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

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.datapath, self.image[idx]))
        landmark = self.landmark[idx]
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(landmark).float()


def get_dataloader(train_transform, valid_transform, batch):
    data_path = os.path.join(os.path.curdir, "data")
    train_path = os.path.join(data_path, "synthetics_train")
    valid_path = os.path.join(data_path, "aflw_val")

    with open(os.path.join(train_path, "annot.pkl"), "rb") as f:
        train_obj = pickle.load(f)
    with open(os.path.join(valid_path, "annot.pkl"), "rb") as f:
        valid_obj = pickle.load(f)
        
    train_img, train_landmark = train_obj
    valid_img, valid_landmark = valid_obj

    train_set = myDataset(train_path, train_img, train_landmark, train_transform)
    valid_set = myDataset(valid_path, valid_img, valid_landmark, valid_transform)

    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch, shuffle=False)

    return train_loader, valid_loader
