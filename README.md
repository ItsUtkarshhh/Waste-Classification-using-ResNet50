# Waste Classification using ResNet50 <br>
A deep learning-based image classification model to classify different types of waste, trained using the Trashnet dataset. This project is built using PyTorch and can be used to predict the category of waste items from images.

## Table of Contents <br>
- Project Overview
- Features
- Tech Stack
- Setup Guide
  - Fork and Clone
  - Setting Up Environment
  - Downloading the Dataset
  - Running the Model
  - Evaluating the Model
- Contributing
- License

## Project Overview
The Waste Classification Model helps in classifying waste into different categories such as paper, plastic, glass, etc., using a deep learning model. It is trained using the Trashnet dataset and implements evaluation metrics such as Accuracy, Precision, Recall, F1 Score and Confusion Matrix.

## Features
- Multi-class classification of waste images.
- PyTorch based training and inference.
- Supports metrics like Accuracy, Precision, Recall, and F1 Score.
- Pretrained Model can be loaded and used for inference on new images.

## Tech Stack
- Tech Stack
- Python 3.8+
- PyTorch
- Scikit-Learn (for metrics)
- Matplotlib (for plotting)
- Jupyter Notebooks (for experimentation)

## Setup Guide

### 1. Fork and Clone
To get started with this project, you need to fork the repository and clone it to your local machine.

#### Step 1: Fork the Repository
Go to the top right of this repository and click the **Fork** button. This will create a copy of the repository under your GitHub account.

#### Step 2: Clone the Repository
Once the repository is forked, clone it locally by running the following commands:

```bash
git clone https://github.com/your-username/waste-classification-model.git
cd waste-classification-model
```

### 2. Setting up the Environment
You need to set up the environment with the required dependencies. It’s recommended to use a virtual environment (either venv or conda).

#### Using `venv`
1. Create a virtual environment:
```bash
python -m venv venv
```
2. Activate the virtual environment:
- On Windows:
```bash
venv\Scripts\activate
```
- On MacOS/Linux:
```bash
source venv/bin/activate
```
3. Install the required dependencies:
```bash
pip install -r requirements.txt
```
Alternatively, if there’s no requirements.txt, you can manually install the dependencies:
```bash
pip install torch torchvision matplotlib scikit-learn
```

### 2. Downloading the Dataset
You will need the Trashnet dataset for training and testing. You can download it from Kaggle:
1. Download the Trashnet dataset from Kaggle: TrashNet Dataset
2. Extract the dataset and place it inside the dataset folder of this repository.
