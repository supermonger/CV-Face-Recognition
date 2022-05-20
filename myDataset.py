import json
from torch.utils.data import DataLoader, Dataset
import PIL.Image as Image
import torch

class myDataset(Dataset):
    def __init__(self, images, coordinates, transform = None):
        self.image = images
        self.landmark = coordinates
        self.transform = transform
    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, idx):
        # print(self.image[idx])
        img = Image.open(self.image[idx])
        landmark = self.landmark[idx]
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(landmark).float()

def get_dataloader(train_transform = None, valid_transform = None):
    with open("data.json", 'r') as f:
        data_obj = json.load(f)

    all_images, all_landmarks = data_obj["images"], data_obj["landmark_localization"]
    N = int(len(all_images) * 0.7)
    train_images = all_images[:N]
    train_landmarks = all_landmarks[:N]

    valid_images = all_images[N:]
    valid_landmarks = all_landmarks[N:]
    train_set = myDataset(train_images, train_landmarks, train_transform)
    valid_set = myDataset(valid_images, valid_landmarks, valid_transform)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=16, shuffle=False)

    return train_loader, valid_loader




