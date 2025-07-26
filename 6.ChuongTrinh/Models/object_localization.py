import torch
from torch import nn
from torchvision import models


class ResNet50BBox(nn.Module):
    def __init__(self):
        super(ResNet50BBox, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(2048, 8)  # Dự đoán 8 giá trị bounding box

    def forward(self, x):
        x = self.resnet(x)
        return torch.sigmoid(x)