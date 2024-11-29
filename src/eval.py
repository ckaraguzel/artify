import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from torchvision import models
from config import Config
from data_loader import PaintingDataLoader
import pandas as pd
import os
from model import get_model
import matplotlib.pyplot as plt
import seaborn as sns
from save_results import SaveResults

# Function to evaluate the model
def evaluate_model(model, data_loader, class_names):
    model.eval()
    all_preds, all_labels = [], []
    val_loss = 0
    criterion = torch.nn.CrossEntropyLoss()  # Assuming classification problem with CrossEntropy

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0, output_dict=True)

    # Calculate accuracy
    val_accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100

    # Print the report for the terminal
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
   
    # Save classification report and confusion matrix
    save_results = SaveResults(base_path=f'C://Users//Hatice//Documents//GitHub//artify//results',model_name=Config.MODEL_NAME)  
    save_results.save_classification_report_to_csv("evaluation_report.csv", report_dict)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    save_results.save_confusion_matrix_as_png("confusion_matrix.png", conf_matrix, class_names)
    save_results.save_csv_as_png("evaluation_report.csv", "evaluation_report.png")

    return val_accuracy, val_loss, precision, recall, f1  # Return validation accuracy and loss

# Load the model
config = Config()
model = get_model(config)  # Ensure the architecture matches the saved model
checkpoint_path = config.MODEL_SAVE_PATH

# Load the saved state_dict
model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
model.to(config.DEVICE)

# Instantiate the PaintingDataLoader object
data_loader = PaintingDataLoader(Config)
_, _, test_loader = data_loader.get_dataloaders()

# Use the actual class names from the full dataset for the report
class_names = data_loader.full_dataset.classes  # Get class names from ImageFolder

# Evaluate the model on the test set
val_accuracy, val_loss, precision, recall, f1 = evaluate_model(model, test_loader, class_names)
