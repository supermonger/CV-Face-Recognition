from dataclasses import dataclass
import torch
from myDataset import get_dataloader
from torchvision.transforms import transforms, autoaugment
from myModel import get_model, WingLoss
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import time
import math
import random
import cv2
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd

def cal_NMEloss(ldmk, pts68_gt):
    ldmk = ldmk.detach()
    pts68_gt = pts68_gt.detach()
    dis = ldmk - pts68_gt
    dis = torch.sqrt(torch.sum(torch.pow(dis, 2), 2))
    dis = torch.mean(dis, axis=1)
    x = dis / 384
    return torch.sum(x, axis=0).item()


def preprocessing(data_path, batch=64, num_workers=8):
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(), 
            transforms.Lambda(lambda x: cv2.GaussianBlur(np.array(x), ksize = (5, 5), sigmaX=0)),
            transforms.ToPILImage(),
            autoaugment.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    )
    valid_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(), 
            transforms.Lambda(lambda x: cv2.GaussianBlur(np.array(x), ksize = (5, 5), sigmaX=0)),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    )
    # train_loader, valid_loader = get_dataloader(train_transform, valid_transform, batch)
    train_loader, valid_loader, test_loader = get_dataloader(data_path, train_transform, valid_transform, batch, num_workers)

    return train_loader, valid_loader, test_loader


def train(train_loader, valid_loader, checkpoint):
    model = get_model(num_classes=68 * 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epoch = 250
    best_NME = 10**3
    loss_fn = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=num_epoch // 10, t_total=num_epoch)
    model.to(device)
    
    for epoch in range(num_epoch):
        print(f"epoch{epoch}")
        train_loss, valid_loss, train_NME, valid_NME = 0.0, 0.0, 0.0, 0.0
        model.train()

        for img, landmark, weight in tqdm(train_loader):

            img = img.to(device)
            landmark = landmark.to(device)
            weight = weight.to(device)
            out = model(img)
            out = out.reshape(out.shape[0], 68, 2)

            loss = (loss_fn(out, landmark) * (1 + weight)).mean()
            # loss = loss_fn(out, landmark)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_NME += cal_NMEloss(out, landmark)

        scheduler.step()
        train_loss = train_loss / len(train_loader.dataset)
        train_NME = train_NME / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():

            for img, landmark, _ in tqdm(valid_loader):
                img = img.to(device)
                landmark = landmark.to(device)
                out = model(img)
                out = out.reshape(out.shape[0], 68, 2)

                loss = loss_fn(out, landmark)

                valid_loss += loss.item()
                valid_NME += cal_NMEloss(out, landmark)

        valid_loss = valid_loss / len(valid_loader.dataset)
        valid_NME =  valid_NME / len(valid_loader.dataset)

        print(f"train loss is {train_loss} and NME is {train_NME}")
        print(f"validation loss is {valid_loss} and NME is {valid_NME}")
    
        if valid_NME < best_NME:
            best_NME = valid_NME
            print("Save the best model !")
            # torch.save(model.state_dict(), "best_mnasnet0_75_batch_32*2_blur_0.001_wing121.pt")
            # torch.save(model.state_dict(), "best.pt")
            torch.save(model.state_dict(), checkpoint)


def prediction(result, batch_idx, batch):
    result_path = os.path.join(os.path.join(os.path.curdir, "data"), "prediction")
    result = result.cpu().detach().numpy()
    if os.path.isdir(result_path) is False:
        os.mkdir(result_path)

    for i in range(result.shape[0]):
        number = (batch_idx * batch) + i
        h = number // 100
        ten = (number - h * 100) // 10
        las = number - h * 100 - ten * 10
        fname = f"pred{h}{ten}{las}.txt"
        with open(os.path.join(result_path, fname), "w") as f:
            for j in range(result.shape[1]):
                f.write(f"{result[i,j,0]} {result[i,j,1]}\n")


def val(checkpoint, valid_loader, batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=68 * 2)
    model.load_state_dict(torch.load(checkpoint))
    model.to(device)

    model.eval()
    with torch.no_grad():
        for batch_idx, (img, landmark) in enumerate(tqdm(valid_loader)):
            img = img.to(device)
            landmark = landmark.to(device)
            out = model(img)
            out = out.reshape(out.shape[0], 68, 2)
            prediction(out, batch_idx, batch)

def test(checkpoint, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=68 * 2)
    model.load_state_dict(torch.load(checkpoint, map_location='cuda:0'))
    model.to(device)
    output_dict = {}

    model.eval()
    with torch.no_grad():
        for fnames, img in tqdm(test_loader):
            img = img.to(device)
            outs = model(img).cpu().detach().numpy()
            for fname, out in zip(fnames, outs):
                output_dict[fname] = out
    pd.DataFrame(output_dict).T.to_csv('solution.txt', header=False, sep=" ")

def fixed_seed(myseed):
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)

class WarmupCosineSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))