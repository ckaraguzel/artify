# model.py
import torch
import torch.nn as nn
from torchvision import models
from config import Config

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.model = Config.MODEL(pretrained=True)  
        if isinstance(self.model, models.VGG):  # If the model is a VGG variant
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_features, num_classes)
        elif isinstance(self.model, models.DenseNet):  # If the model is a DenseNet variant
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_features, num_classes)
        else:  # For models like ResNet
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)

def get_model(config):
    model = Classifier(config.NUM_CLASSES)
    return model.to(config.DEVICE)

