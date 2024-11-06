import os
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from config import DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT

# Define transformations
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor()
])

def load_data():
    # Load the dataset
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

    # Calculate split sizes
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = int(VAL_SPLIT * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Split the dataset
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, len(dataset.classes), dataset.classes
