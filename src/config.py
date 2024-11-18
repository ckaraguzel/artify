# Define constants and parameters for the project
import torch
class Config:
    # Dataset and paths
    DATA_DIR = 'C://Users//Hatice//Documents//GitHub//artify//data//processed_data'
    IMG_WIDTH= 224
    IMG_HEIGHT=224
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 25
    MODEL_SAVE_PATH = 'saved_models//resnet_model_data_aug_early_stop.pth'
    MODEL_NAME= "resnet_model_data_aug_early_stop"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 7
    # Train/Validation/Test split ratios
    TRAIN_RATIO = 0.3
    VAL_RATIO = 0.1
    TEST_RATIO = 0.6  # Remaining portion for test data

    # Class weights (will be calculated based on dataset)
    CLASS_WEIGHTS = None
