# 🎨 Artify: Finding the painter of a Masterpiece with CNNs 🖌️

Welcome to Artify, where Convolutional Neural Networks (CNNs) bring the genius of art to life! 🧑‍🎨✨ Can AI tell the difference between Picasso's bold geometry and and Dalí's surreal dreamscapes? Spoiler alert: it absolutely can!

This project trains a CNN to dive deep into the textures, colors, and brushwork of famous paintings, unraveling the secrets of each artist’s unique style. Forget genres—this is pure artistic detective work powered by cutting-edge machine learning.

Ready to let AI channel its inner art historian? Let’s create something extraordinary! 🚀

# Overview

We fine-tuned several pre-trained Convolutional Neural Network (CNN) models (resnet18, densenet121) to classify paintings based on their painters. This project does not involve genre classification and exclusively focuses on the identification of the painter. We used a dataset of paintings of 7 painters (Claude Monet, Georges Braque, Pablo Picasso, Paul Cezanne, Pierre-August Renoir, Salvador Dalí, Vincent Van Gogh) and hence our model can only provide more accurate results for the paintings of these particular painters. We created an application [Artify_App](https://huggingface.co/spaces/hmutlu/Artify) based on our best performing model. 

# Dataset

The dataset comprises a collection of paintings by notable artists across different art movements. Each image is labeled with the painter's name, forming the basis for the supervised learning model. We downloaded the data from: https://www.kaggle.com/datasets/antoinegruson/-wikiart-all-images-120k-link

# Preprocessing Steps:

- Standardizing image dimensions through resizing

- Normalizing pixel values for consistent model input

- Splitting the dataset into training, validation, and test subsets

- On the training set: normalization, augmentation (e.g., random crops, flips, and rotations), resize

- On the test and validation sets: normalization and resize

# Dependencies

```text
torch, torchvision, streamlit, pillow, tqdm, scikit-learn, pandas, numpy, matplotlib 
```

# Fine-tuned Models and Results

- ResNet18
  
  The model achieves the following performance metrics on the test set:
 ![resnet18_optuna](https://github.com/user-attachments/assets/5dfc8c70-2f72-492d-a498-a9f12d55c33d)

  
   
- DenseNet121
  
  The model achieves the following performance metrics on the test set:
  ![densenet121_optuna_](https://github.com/user-attachments/assets/823b96e7-5693-47dc-9066-356eeb531007)



# Application

[Artify_App](https://huggingface.co/spaces/hmutlu/Artify)
 
<img width="579" alt="application_screenshot" src="https://github.com/user-attachments/assets/be71c59b-466b-4d62-8452-67ad249db6cf">





