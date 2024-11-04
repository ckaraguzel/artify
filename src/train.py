from data_preprocessing import load_data, count_paintings
from model import create_model
from config import EPOCHS, MODEL_SAVE_PATH

# Count paintings per painter
painter_counts = count_paintings()
print("Number of paintings per painter:")
for painter, count in painter_counts.items():
    print(f"{painter}: {count}")

# Load datasets
train_ds, val_ds, test_ds = load_data()

# Get the number of classes from the dataset
num_classes = len(train_ds.class_names)

# Create model
model = create_model(num_classes)

# Train the model
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# Save the model
model.save(MODEL_SAVE_PATH)
