import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from config import Config

class PaintingDataLoader:
    def __init__(self, config):
        self.config = config
        # Data Loader

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop((224,224)), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])

        self.validation_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
        # Load the dataset with ImageFolder
        self.full_dataset = datasets.ImageFolder(self.config.DATA_DIR)
        self.targets = np.array(self.full_dataset.targets)  # Class labels for stratification

        # Perform stratified splits for train, validation, and test sets
        self.train_indices, self.val_indices, self.test_indices = self._stratified_split()
        
        # Create Subsets for each split
        self.train_dataset = Subset(self.full_dataset, self.train_indices)
        self.train_dataset.dataset.transform = self.train_transform  # Apply augmentation only to training set


        self.val_dataset = Subset(self.full_dataset, self.val_indices)
        self.val_dataset.dataset.transform = self.validation_transform  # No augmentation for validation set


        self.test_dataset = Subset(self.full_dataset, self.test_indices)
        self.test_dataset.dataset.transform = self.validation_transform  # No augmentation for test set


    def _stratified_split(self):
        """Perform a stratified split to obtain train, validation, and test indices."""
        # Initial train/test split with stratification
        train_indices, temp_indices = train_test_split(
            np.arange(len(self.targets)),
            test_size=(1 - self.config.TRAIN_RATIO),
            stratify=self.targets,
            random_state=42
        )
        
        # Further split temp set into validation and test sets with stratification
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=self.config.TEST_RATIO / (self.config.TEST_RATIO + self.config.VAL_RATIO),
            stratify=self.targets[temp_indices],
            random_state=42
        )
        
        return train_indices, val_indices, test_indices

    def get_dataloaders(self):
        """Return DataLoaders for train, validation, and test sets."""
        train_loader = DataLoader(self.train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        return train_loader, val_loader, test_loader
