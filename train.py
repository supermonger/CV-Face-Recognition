from tkinter import Image
import torch
from myDataset import get_dataloader
from torchvision import transforms
from myModel import get_model
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os


def cal_NMEloss(ldmk, pts68_gt):
    ldmk = ldmk.cpu().detach().numpy()
    pts68_gt = pts68_gt.cpu().detach().numpy()
    dis = ldmk - pts68_gt
    dis = np.sqrt(np.sum(np.power(dis, 2), 2))
    dis = np.mean(dis, axis=1)
    x = dis / 384
    return np.sum(x, axis=0)


def preprocessing(batch=64):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            # transforms.RandomRotation(30),
            # transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
            # transforms.Grayscale() paper have used, need to change model channel
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    valid_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train_loader, valid_loader = get_dataloader(train_transform, valid_transform, batch)

    return train_loader, valid_loader


def train(train_loader, valid_loader):

    model = get_model(num_classes=68 * 2)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    num_epoch = 100
    model.to(device)
    model.train()
    for epoch in range(num_epoch):
        print(f"epoch{epoch}")
        train_loss = 0.0
        valid_loss = 0.0
        train_NME = 0.0
        valid_NME = 0.0
        for i, (img, landmark) in enumerate(tqdm(train_loader)):

            img = img.to(device)
            landmark = landmark.to(device)
            out = model(img)
            out = out.reshape(out.shape[0], 68, 2)

            loss = loss_fn(out, landmark)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            train_NME += cal_NMEloss(out, landmark)

        train_loss = train_loss / len(train_loader.dataset)
        train_NME = train_NME / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            for i, (img, landmark) in enumerate(tqdm(valid_loader)):
                img = img.to(device)
                landmark = landmark.to(device)
                out = model(img)
                out = out.reshape(out.shape[0], 68, 2)

                loss = loss_fn(out, landmark)

                valid_loss += loss.item()
                valid_NME += cal_NMEloss(out, landmark)

        valid_loss = valid_loss / len(valid_loader.dataset)
        valid_NME = valid_NME / len(valid_loader.dataset)

        print(f"train loss is {train_loss} and NME is {train_NME}")
        print(f"validation loss is {valid_loss} and NME is {valid_NME}")

    torch.save(model.state_dict(), "best.pt")


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


def test(checkpoint, valid_loader, batch):
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


if __name__ == "__main__":
    checkpoint = "best.pt"
    batch = 16
    train_loader, valid_loader = preprocessing(batch)
    train(train_loader, valid_loader)
    # test(checkpoint, valid_loader, batch)
