import torch
from myDataset import get_dataloader
from torchvision import transforms
from myModel import get_model
import torch.nn as nn
import torch.optim as optim 
from tqdm import tqdm
import numpy as np

def cal_NMEloss(ldmk, pts68_gt):
    ldmk = ldmk.cpu().detach().numpy()
    pts68_gt = pts68_gt.cpu().detach().numpy()
    dis = (ldmk - pts68_gt)
    dis = np.sqrt(np.sum(np.power(dis, 2), 1))
    dis = np.mean(dis)
    x = dis / 384
    return x

def train():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
                                        transforms.RandomRotation(30),
                                        transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
                                        # transforms.Grayscale() paper have used, need to change model channel
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std),
                                        
                                        ])
    valid_transform = transforms.Compose([
                                        
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std),
                                        ])
    batch = 64
    train_loader, valid_loader = get_dataloader(train_transform, valid_transform, batch)
    model = get_model(num_classes=68*2)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            train_NME = cal_NMEloss(out, landmark)

        train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            for i, (img, landmark) in enumerate(tqdm(valid_loader)):
                img = img.to(device)
                landmark = landmark.to(device)
                out = model(img)
                out = out.reshape(out.shape[0], 68, 2)

                loss = loss_fn(out, landmark)

                valid_loss += loss.item()
                valid_NME = cal_NMEloss(out, landmark)

        valid_loss = valid_loss / len(train_loader.dataset)

        print(f"train loss is {train_loss} and NME is {train_NME}")
        print(f"validation loss is {valid_loss} and NME is {valid_NME}")

    torch.save(model.state_dict(), "best.pt")

if __name__ == "__main__":
    train()
    