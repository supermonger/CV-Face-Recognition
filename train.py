import torch
from myDataset import get_dataloader
from torchvision import transforms
from myModel import get_model
import torch.nn as nn
import torch.optim as optim 
from tqdm import tqdm

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

    train_loader, valid_loader = get_dataloader(train_transform, valid_transform)
    model = get_model(num_classes=70*2)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epoch = 10
    model.to(device)
    model.train()
    for epoch in range(num_epoch):
        print(f"epoch{epoch}")
        train_loss = 0.0
        valid_loss = 0.0
        for i, (img, landmark) in enumerate(tqdm(train_loader)):

            img = img.to(device)
            landmark = landmark.to(device)
            out = model(img)

            loss = loss_fn(out, landmark)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        with torch.no_grad:
            for i, (img, landmark) in enumerate(tqdm(valid_loader)):
                img = img.to(device)
                landmark = landmark.to(device)
                out = model(img)

                loss = (out, landmark)

                valid_loss += loss.item()

        valid_loss = valid_loss / len(train_loader.dataset)

        print(f"train loss is {train_loss}")
        print(f"validation loss is {valid_loss}")

    torch.save(model.state_dict(), "best.pt")

if __name__ == "__main__":
    train()
    