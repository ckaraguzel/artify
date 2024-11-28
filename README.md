# 🎨 Artify: Finding the painter of a Masterpiece with CNNs 🖌️

Welcome to Artify, where Convolutional Neural Networks (CNNs) bring the genius of art to life! 🧑‍🎨✨ Can AI tell the difference between Picasso's bold geometry and and Dalí's surreal dreamscapes? Spoiler alert: it absolutely can!

This project trains a CNN to dive deep into the textures, colors, and brushwork of famous paintings, unraveling the secrets of each artist’s unique style. Forget genres—this is pure artistic detective work powered by cutting-edge machine learning.

Ready to let AI channel its inner art historian? Let’s create something extraordinary! 🚀

# Overview

We fine-tuned several pre-trained Convolutional Neural Network (CNN) models (resnet18, densenet121) to classify paintings based on their painters. This project does not involve genre classification and exclusively focuses on the identification of the painter. 

We created an application [Artify_App](https://huggingface.co/spaces/hmutlu/Artify) based on our best performing model. 

# Dataset
The dataset comprises a collection of paintings by notable artists across different art movements. Each image is labeled with the painter's name, forming the basis for the supervised learning model. We downloaded the data from: https://www.kaggle.com/datasets/antoinegruson/-wikiart-all-images-120k-link

# Preprocessing Steps:

- Standardizing image dimensions through resizing

- Normalizing pixel values for consistent model input

- Splitting the dataset into training, validation, and test subsets

- On the training set: Normalization, augmentation (e.g., random crops, flips, and rotations), resize

- On the test and validation sets: Normalization and resize

# Painters
The model predicts the painter from the following list of renowned artists:

Claude Monet

Pierre-Auguste Renoir

Vincent van Gogh

Paul Cézanne

Pablo Picasso

Georges Braque

Salvador Dalí



# Model Architecture
The painter classification model leverages a deep learning framework to identify unique features of each painter's style.


## 3Key features:

Pretrained Backbone: ResNet18, Densenet121

Transfer Learning: Fine-tuned for painter-specific classification

Evaluation Metrics: Includes accuracy, precision, recall, and F1-score

# Results
The model achieves the following performance metrics on the test set:

Accuracy: XX%

Precision: XX%

Recall: XX%

F1-Score: XX%

Prediction examples and performance visualizations can be found in the results/ folder.



