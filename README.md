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
  1. The training and validation loss and accuracies per epoch are as follows:
  ![resnet18_training](https://github.com/user-attachments/assets/1b31e6d2-c14c-4466-aa11-326cce19ce06)

  
  2. The model achieves the following performance metrics on the test set:
  ![R_evaluation_report](https://github.com/user-attachments/assets/f132f972-4409-4cda-9e77-3b621bceff39)

  3. Confusion matrix: 
  ![resnet18_optuna_confusion_matrix](https://github.com/user-attachments/assets/d6fe682a-6296-4ed9-af5b-f7f8471f234a)  
   
- DenseNet121
  1. The training and validation loss and accuracies per epoch are as follows:
  ![densenet121_training](https://github.com/user-attachments/assets/ff0397ba-20ec-44aa-b828-2e739190cb7e)
  
  2. The model achieves the following performance metrics on the test set:
  ![D_evaluation_report](https://github.com/user-attachments/assets/fa939f56-de9b-49d3-b50b-96068dfc2fb9)

  3. Confusion matrix:
  ![densenet121_confusion_matrix](https://github.com/user-attachments/assets/729a4637-a91b-423e-9fb6-8b31a2746b36)


# Application

[Artify_App](https://huggingface.co/spaces/hmutlu/Artify)
 
<img width="579" alt="application_screenshot" src="https://github.com/user-attachments/assets/be71c59b-466b-4d62-8452-67ad249db6cf">





