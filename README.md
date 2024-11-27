# Artify: Painter Classification using Machine Learning

Artify is a machine learning project designed to predict the painter of a given painting. By analyzing the unique artistic styles of renowned painters, the model identifies the creator without considering the genre of the painting.

# Overview
The goal of Artify is to classify paintings based on their painters, using machine learning techniques to capture the nuances of artistic style. The project focuses exclusively on identifying the painter and does not involve genre classification.

# Dataset
The dataset comprises a collection of paintings by notable artists across different art movements. Each image is labeled with the painter's name, forming the basis for the supervised learning model. We downloaded the data from: https://www.kaggle.com/datasets/antoinegruson/-wikiart-all-images-120k-link

# Preprocessing Steps:

Standardizing image dimensions through resizing
Normalizing pixel values for consistent model input
Splitting the dataset into training, validation, and test subsets

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

Pretrained Backbone: ResNet

Transfer Learning: Fine-tuned for painter-specific classification

Evaluation Metrics: Includes accuracy, precision, recall, and F1-score

# Results
The model achieves the following performance metrics on the test set:

Accuracy: XX%

Precision: XX%

Recall: XX%

F1-Score: XX%

Prediction examples and performance visualizations can be found in the results/ folder.



