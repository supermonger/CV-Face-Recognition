import torch.nn as nn
from torchvision.models import resnet34

class myModel(nn.Module):
    def __init__(self, num_classes = 2):
        super().__init__()
        self.resnet = resnet34(pretrained=True)
        self.fc1 = nn.Sequential(nn.Linear(self.resnet.fc.out_features, num_classes))

    def forward(self, x):
        out = self.resnet(x)
        out = self.fc1(out)
        return out

def get_model(num_classes):
    model = myModel(num_classes)
    return model