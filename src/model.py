# model.py
import torch
import torch.nn as nn
from torchvision import models

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)  # Using ResNet18; can be switched to ResNet50
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)

def get_model(config):
    model = ResNetClassifier(config.NUM_CLASSES)
    return model.to(config.DEVICE)

