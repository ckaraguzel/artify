import tensorflow as tf
from sklearn.model_selection import train_test_split
from config import DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT
import os
from collections import Counter

def load_data():
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=123
    )

    # Split dataset into train, validation, and test sets
    train_size = int(len(dataset) * TRAIN_SPLIT)
    val_size = int(len(dataset) * VAL_SPLIT)
    test_size = len(dataset) - train_size - val_size

    train_ds = dataset.take(train_size)
    test_val_ds = dataset.skip(train_size)
    val_ds = test_val_ds.take(val_size)
    test_ds = test_val_ds.skip(val_size)

    return train_ds, val_ds, test_ds

def count_paintings():
    painter_counts = Counter()
    for painter_folder in os.listdir(DATA_DIR):
        painter_path = os.path.join(DATA_DIR, painter_folder)
        if os.path.isdir(painter_path):
            painting_count = len(os.listdir(painter_path))
            painter_counts[painter_folder] = painting_count
    return painter_counts