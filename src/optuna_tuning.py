import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model  # Assuming this function gets the model architecture
from data_loader import PaintingDataLoader  # Assuming this loads your dataset
from config import Config
import os
from save_results import SaveResults
from tqdm import tqdm


def calculate_class_weights(data_loader):
    label_counts = torch.zeros(Config.NUM_CLASSES)
    for _, labels in data_loader:
        for label in labels:
            label_counts[label] += 1
    weights = 1.0 / label_counts
    return weights


# Define the objective function for Optuna
def objective(trial):
    # Define hyperparameters to tune
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])  # Tune batch size
    epochs = trial.suggest_int('epochs', 3, 10)  # Tune number of epochs (3 to 10)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)  # Tune learning rate
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)  # Tune weight decay for regularization
    
    # Update config with current trial's hyperparameters
    config = Config()
    config.batch_size = batch_size
    config.epochs = epochs
    config.learning_rate = learning_rate
    config.weight_decay = weight_decay
    
    # Load data with the current batch size
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
    
    # Train the model for the given number of epochs
    for epoch in range(config.epochs):
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
        print(f"Epoch [{epoch+1}/{config.epochs}], Train Loss: {epoch_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")
    
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
        print(f"Epoch [{epoch+1}/{config.epochs}], Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Saving the best model checkpoint based on validation accuracy
        model_dir = f'C://Users//Hatice//Documents//GitHub//artify//saved_models//optuna_model_check_points//model_checkpoints_{config.MODEL_NAME}'  # Directory where the models will be saved
        os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist
        model_path = os.path.join(model_dir, f"best_model_{trial.number}.pth")  # Model file path with trial number
        torch.save(model.state_dict(), model_path)  # Save the model's state_dict

    return val_accuracy  # Optuna will maximize this metric

# Create an Optuna study
study = optuna.create_study(direction='maximize')  # We want to maximize validation accuracy
study.optimize(objective, n_trials=10)  # Number of trials (configurations) to test

# Save the best configuration
best_params = study.best_params
print("Best hyperparameters found: ", best_params)