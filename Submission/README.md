# Project Title

## Environment Setup

To set up your environment, please follow these steps:

1. Create the Conda environment:
   ```bash
   conda env create -f environment.yml
   # Or for Mac users
   conda env create -f mac_environment.yml
   ```
2. conda activate env_name

## Dataset

Download the CelebA dataset and place the images in the img_align_celeba folder:
CelebA Dataset on Kaggle https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/

There are 4 notebooks.

## Data Preprocessing

Data-Preprocessing.ipynb: Contains the preprocessing steps to obtain processed_img. This notebook explains the development of the processing script and visualizations of the results. The last cell will run the processing functions from data_preprocessing.py.

## Model Training

Basic CNN.ipynb: This notebook runs both the processed and unprocessed images through a basic CNN model for gender classification.
ResNet Transfer Learning.ipynb: Demonstrates transfer learning from ResNet50, tailored for gender classification.
Multi Attribute ResNet.ipynb: Improves upon the previous model by incorporating additional attribute information provided in the CelebA dataset.
