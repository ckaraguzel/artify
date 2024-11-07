import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torchvision import models
from config import Config
from data_loader import PaintingDataLoader
from torchvision.models import ResNet18_Weights

# Load the model and configure the final layer
model = models.resnet18(weights=None)  # No pretrained weights
num_classes = 7  # Adjust based on your dataset
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load the state_dict, ensuring weights_only=True to avoid warnings
state_dict = torch.load("C://Users//Hatice//Documents//GitHub//artify//saved_models//resnet_model.pth", weights_only=True)
state_dict.pop('fc.weight', None)
state_dict.pop('fc.bias', None)
model.load_state_dict(state_dict, strict=False)  # strict=False allows missing keys like 'fc'
model.eval()  # Set model to evaluation mode

# Set device (CPU or GPU)
device = torch.device('cpu')  # Change to 'cuda' if you want to use a GPU

# Function to evaluate the model
def evaluate_model(model, data_loader, class_names):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    # Disable gradient computation to speed up inference
    with torch.no_grad():
        for images, labels in data_loader:
            # Move images and labels to the correct device (CPU or GPU)
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass through the model
            outputs = model(images)
            
            # Get predicted labels (highest logit)
            _, preds = torch.max(outputs, 1)
            
            # Collect the predictions and true labels
            all_preds.extend(preds.cpu().numpy())  # Move back to CPU for metric calculations
            all_labels.extend(labels.cpu().numpy())  # Move back to CPU for metric calculations

    # Calculate precision, recall, and F1 for each class using sklearn metrics
    precision = precision_score(all_labels, all_preds, average=None, labels=np.unique(all_labels),zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, labels=np.unique(all_labels), zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None, labels=np.unique(all_labels), zero_division=0)

    # Print per-class metrics
    print(f"Precision per class: {precision}")
    print(f"Recall per class: {recall}")
    print(f"F1 Score per class: {f1}")

    # Print classification report with class names
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    return precision, recall, f1

# Instantiate the PaintingDataLoader object
data_loader = PaintingDataLoader(Config)
_, _, test_loader = data_loader.get_dataloaders()

# Use the actual class names from the full dataset for the report
class_names = data_loader.full_dataset.classes  # Get class names from ImageFolder

# Evaluate the model on the test set
precision, recall, f1 = evaluate_model(model, test_loader, class_names)
