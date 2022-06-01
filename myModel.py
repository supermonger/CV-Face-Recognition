import torch.nn as nn
from torchvision.models import resnet34, mobilenet_v2, mobilenet_v3_small


class myMobileNet_V3(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.mobilenet = mobilenet_v3_small(pretrained=True)
        self.fc1 = nn.Sequential(
            nn.Linear(self.mobilenet.classifier[3].out_features, 512),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
        )
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.mobilenet(x)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out


class myMobileNet_V2(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.mobilenet = mobilenet_v2(pretrained=True)
        self.fc1 = nn.Sequential(
            nn.Linear(self.mobilenet.classifier[1].out_features, 512),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.mobilenet(x)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out


def get_model(num_classes):
    # model = myModel(num_classes)
    model = myMobileNet_V3(num_classes)
    return model
