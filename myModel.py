import torch.nn as nn
from torchvision.models import mnasnet0_75
import torch
import math

class myMnasnet0_75(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.mnasnet = mnasnet0_75(pretrained=False)
        # self.mnasnet.classifier[1].out_features = num_classes
        self.mnasnet.layers[0] = nn.Sequential(nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False))
        self.mnasnet.classifier[1] = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 512))
        
        self.fc_layers = nn.Sequential(
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.mnasnet(x)
        out = self.fc_layers(x)
        return out


def get_model(num_classes):
    model = myMnasnet0_75(num_classes)

    return model

class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))