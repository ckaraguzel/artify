import torch
from data_preprocessing import load_data
from model import CNNModel
from config import MODEL_SAVE_PATH

# Load data and model
_, _, test_loader, num_classes, class_names = load_data()
model = CNNModel(num_classes)
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

# Evaluation
test_loss = 0.0
correct = 0
total = 0
criterion = nn.CrossEntropyLoss()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total
print(f"Test Loss: {test_loss / len(test_loader)}, Test Accuracy: {test_accuracy}%")
