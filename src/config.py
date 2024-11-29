# Define constants and parameters for the project
#config
import torch
from torchvision import models
class Config:
    # Dataset and paths
    DATA_DIR = 'C://Users//Hatice//Documents//GitHub//artify//data//processed_data'
    IMG_WIDTH= 224
    IMG_HEIGHT=224
    
    LEARNING_RATE = 1.0420672283470088e-05  # resnet18
    WEIGHT_DECAY = 0.0009609681530757624   # resnet18
    BATCH_SIZE = 32       # resnet18
    NUM_EPOCHS = 10       # resnet18
    #LEARNING_RATE = 1.044905336175204e-05  # densenet121
    #WEIGHT_DECAY = 0.00030393535178517415   # densenet121
    #BATCH_SIZE = 32       # densenet121
    #NUM_EPOCHS = 10       # densenet121
    MODEL_SAVE_PATH = 'saved_models//resnet18_optuna.pth'
    MODEL= models.resnet18
    MODEL_NAME= "resnet18_optuna"
    RESULTS_SAVE_PATH='results//resnet18_optuna'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 7
    # Train/Validation/Test split ratios
    TRAIN_RATIO = 0.3
    VAL_RATIO = 0.1
    TEST_RATIO = 0.6  # Remaining portion for test data

    # Class weights (will be calculated based on dataset)
    CLASS_WEIGHTS = None
