# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from data_loader import PaintingDataLoader
from model import get_model
from tqdm import tqdm
import os
import csv
import matplotlib.pyplot as plt
from save_results import SaveResults

def calculate_class_weights(data_loader):
    label_counts = torch.zeros(Config.NUM_CLASSES)
    for _, labels in data_loader:
        for label in labels:
            label_counts[label] += 1
    weights = 1.0 / label_counts
    return weights


def train(config):
    save_results = SaveResults("C://Users//Hatice//Documents//GitHub//artify//results", config.MODEL_NAME)

    # Initialize the PaintingDataLoader and get dataloaders
    data_loader = PaintingDataLoader(config)
    train_loader, val_loader, _ = data_loader.get_dataloaders()
    
    # Initialize model, criterion, and optimizer
    model = get_model(config)
    class_weights = calculate_class_weights(train_loader).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # Initialize lists to store training and validation metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        epoch_loss, correct, total = 0, 0, 0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}'):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        train_losses.append(epoch_loss / len(train_loader))
        train_accuracies.append(train_accuracy)
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], Train Loss: {epoch_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%")
        # Save the results for this epoch to CSV
        save_results.save_results_to_csv(
            f"training_results_{config.MODEL_NAME}.csv",
            ["Phase", "Epoch", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy", "Model Name"],
            ["Training", epoch + 1, epoch_loss / len(train_loader), train_accuracy, val_loss / len(val_loader), val_accuracy, config.MODEL_NAME]
        )

    # Save the model
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model saved to {config.MODEL_SAVE_PATH}")

    # Save the final graph after all epochs
    save_results.plot_and_save_graphs(
        "training_progress.png", train_losses, val_losses, train_accuracies, val_accuracies
    )

if __name__ == '__main__':
    config = Config()
    train(config)
