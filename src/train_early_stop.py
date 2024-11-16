import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from data_loader import PaintingDataLoader
from model import get_model
from tqdm import tqdm
import os
import csv

def calculate_class_weights(data_loader):
    label_counts = torch.zeros(Config.NUM_CLASSES)
    for _, labels in data_loader:
        for label in labels:
            label_counts[label] += 1
    weights = 1.0 / label_counts
    return weights
def save_results_to_csv(file_path, data):
    # Check if the file exists; if not, create it and add headers
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Phase", "Epoch", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy", "Model Name"])
        writer.writerow(data)

def train(config):
    # Initialize the PaintingDataLoader and get dataloaders
    data_loader = PaintingDataLoader(config)
    train_loader, val_loader, _ = data_loader.get_dataloaders()
    
    # Initialize model, criterion, and optimizer
    model = get_model(config)
    class_weights = calculate_class_weights(train_loader).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # Early stopping parameters
    patience = config.PATIENCE  # Number of epochs to wait for improvement
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
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
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        save_results_to_csv(
            "C://Users//Hatice//Desktop//artify 2//results//results.csv",
            ["Training", epoch+1, epoch_loss/len(train_loader), train_accuracy, val_loss/len(val_loader), val_accuracy, config.MODEL_NAME]
        )

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"Validation loss improved. Model saved to {config.MODEL_SAVE_PATH}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print("Early stopping triggered. Training stopped.")
            break

if __name__ == '__main__':
    config = Config()
    config.PATIENCE = 5  # Number of epochs to wait for improvement before stopping
    train(config)