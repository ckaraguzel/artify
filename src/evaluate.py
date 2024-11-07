import torch
import numpy
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, classification_report
from torchvision import models
from config import Config
from data_loader import PaintingDataLoader
from torchvision.models import ResNet18_Weights

# Define the model structure
model = models.resnet18(weights= None)
num_classes = 7  # Adjust this to your dataset's number of classes
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load the model weights
model.load_state_dict(torch.load("C://Users//Hatice//Documents//GitHub//artify//saved_models//resnet_model.pth"))
model.eval()  # Set the model to evaluation mode


test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_loader = PaintingDataLoader(Config)
_, _, test_loader = data_loader.get_dataloaders()

all_preds = []
all_labels = []

# Disable gradient computation for evaluation
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate and print overall accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Accuracy: {accuracy:.4f}")

# Print a classification report for detailed metrics per class
print(classification_report(all_labels, all_preds, target_names=test_loader.classes))
