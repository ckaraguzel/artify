import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from config import Config
from model import get_model



def predict(image_path):
    # Load the model
    config = Config()
    model = get_model(config)  # Ensure the architecture matches the saved model
    checkpoint_path = 'C:/Users/Hatice/Documents/GitHub/artify/saved_models/resnet18_optuna.pth'

    # Load the saved state_dict
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=True))
    model.to(config.DEVICE)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)

    model.eval()
    out = model(batch_t)

    # List of classes (artists)
    classes= ['Claude Monet', 'Georges Braque','Pablo Picasso','Paul Cezanne','Pierre Auguste Renoir','Salvador Dali','Vincent Van Gogh']


    # Get softmax probabilities for each class
    prob = F.softmax(out, dim=1)[0] * 100  # Convert to percentages

    # Create a dictionary of class labels and their probabilities
    class_probs = {classes[i]: prob[i].item() for i in range(len(classes))}

    # Print to debug if the dictionary is being created properly
    print(f"Class probabilities: {class_probs}")

    return class_probs  # Return the dictionary