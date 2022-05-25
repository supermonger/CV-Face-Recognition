import torch.nn as nn
from torchvision.models import mobilenet_v2, mobilenet_v3_small

class ReLU384(nn.Hardtanh):

    def __init__(self, inplace: bool = False):
        super(ReLU384, self).__init__(0., 384., inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class myMobileNet_V3(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.mobilenet = mobilenet_v3_small(pretrained=True)
        self.mobilenet.classifier[3] = nn.Linear(self.mobilenet.classifier[3].in_features, num_classes)
        # self.activator = ReLU384()
        # self.activator = nn.Sigmoid()
        self.activator = nn.Tanh()

    def forward(self, x):
        out = self.mobilenet(x)
        return (self.activator(out) + 1) * 192


class myMobileNet_V2(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.mobilenet = mobilenet_v2(pretrained=True)
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.classifier[1].in_features, num_classes)
        self.activator = ReLU384()

    def forward(self, x):
        out = self.mobilenet(x)
        return self.activator(out)


def get_model(num_classes):
    # model = myModel(num_classes)
    model = myMobileNet_V3(num_classes)
    # print(model)
    return model
