import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from torchvision import models
from config import Config
from data_loader import PaintingDataLoader
import pandas as pd
import os
from model import get_model

# Function to save the classification report to CSV
def save_classification_report_to_csv(file_path, report_dict, model_name):
    """
    Save the classification report to a CSV file.

    Args:
        file_path (str): Path to the CSV file where the report will be saved.
        report_dict (dict): The classification report as a dictionary.
        model_name (str): The name of the model for context in the file.
    """
    # Convert the dictionary to a DataFrame
    report_df = pd.DataFrame(report_dict).transpose()
    report_df["Model Name"] = model_name  # Add model name for context

    # Check if the file exists
    if not os.path.isfile(file_path):
        # Write the DataFrame with headers if the file doesn't exist
        report_df.to_csv(file_path, mode='w', index=True, index_label="Class")
    else:
        # Append to the file without headers
        report_df.to_csv(file_path, mode='a', index=True, index_label="Class", header=False)

# Function to evaluate the model
def evaluate_model(model, data_loader, class_names):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0, output_dict=True)

    # Print the report for the terminal
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    # Save the classification report to a CSV file
    save_classification_report_to_csv(
        "C://Users//Hatice//Documents//GitHub//artify//csv_files//evaluation_report.csv",
        report_dict,
        Config.MODEL_NAME,
    )

    return precision, recall, f1

# Load the model
config = Config()
model = get_model(config)  # Ensure the architecture matches the saved model
checkpoint_path = r"C://Users//Hatice//Documents//GitHub//artify//saved_models//resnet_model_early_stop.pth"

# Load the saved state_dict
model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
model.to(config.DEVICE)

# Instantiate the PaintingDataLoader object
data_loader = PaintingDataLoader(Config)
_, _, test_loader = data_loader.get_dataloaders()

# Use the actual class names from the full dataset for the report
class_names = data_loader.full_dataset.classes  # Get class names from ImageFolder

# Evaluate the model on the test set
precision, recall, f1 = evaluate_model(model, test_loader, class_names)