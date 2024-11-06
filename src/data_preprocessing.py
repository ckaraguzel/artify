'''
This file includes 
-removing non-RGB images
-counting numbers of paintings per painter
-resizing our paintings
'''


import os
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from config import DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT
from PIL import Image
import matplotlib.pyplot as plt

def remove_non_rgb_images(data_dir):
    # Iterate through each painter's folder
    for painter_folder in os.listdir(data_dir):
        painter_path = os.path.join(data_dir, painter_folder)
        
        # Ensure the path is a directory
        if os.path.isdir(painter_path):
            print(f"Checking images in: {painter_folder}")
            
            # Iterate through each image in the painter's folder
            for image_file in os.listdir(painter_path):
                image_path = os.path.join(painter_path, image_file)
                
                try:
                    # Open the image
                    with Image.open(image_path) as img:
                        # Check if image is in RGB format
                        if img.mode != 'RGB':
                            print(f"Removing non-RGB image: {image_path}")
                            os.remove(image_path)  # Remove the image if not RGB
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    # Optionally, remove the corrupted or unreadable file
                    os.remove(image_path)

# Run the function
remove_non_rgb_images(DATA_DIR)

file_counts = {}
for painter_folder in os.listdir(DATA_DIR):
    painter_path = os.path.join(DATA_DIR, painter_folder)
    if os.path.isdir(painter_path):
        num_files = len([f for f in os.listdir(painter_path) if os.path.isfile(os.path.join(painter_path, f))])
        file_counts[painter_folder] = num_files

# Extract data for plotting
painters = list(file_counts.keys())
num_files = list(file_counts.values())

# Create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(painters, num_files)
plt.xlabel("Painter")
plt.ylabel("Number of Files")
plt.title("Number of Images per Painter")
plt.xticks(rotation=45, ha='right')  # Rotate x labels for readability
plt.tight_layout()
plt.show()

transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor()
])

# Define input and output directories
input_folder = '/Users/cisilkaraguzel/Documents/GitHub/artify/data/raw_data'  # Folder containing original images
output_folder = '/Users/cisilkaraguzel/Documents/GitHub/artify/data/processed_data'  # Folder to save processed images

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Traverse through each subfolder in the input directory
for painter_folder in os.listdir(input_folder):
    painter_path = os.path.join(input_folder, painter_folder)
    
    # Only proceed if it's a directory (painter subfolder)
    if os.path.isdir(painter_path):
        # Create corresponding subfolder in the output directory
        processed_painter_path = os.path.join(output_folder, painter_folder)
        os.makedirs(processed_painter_path, exist_ok=True)
        
        # Process each image in the painter's folder
        for filename in os.listdir(painter_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
                img_path = os.path.join(painter_path, filename)
                
                try:
                    # Load image
                    image = Image.open(img_path).convert("RGB")
                    
                    # Apply transformations
                    transformed_image = transform(image)
                    
                    # Convert tensor back to PIL image to save it
                    transformed_image_pil = transforms.ToPILImage()(transformed_image)
                    
                    # Save the processed image in the corresponding painter's subfolder
                    output_path = os.path.join(processed_painter_path, filename)
                    transformed_image_pil.save(output_path)
                    print(f"Processed and saved {filename} to {processed_painter_path}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

print("All images processed and saved successfully.")








