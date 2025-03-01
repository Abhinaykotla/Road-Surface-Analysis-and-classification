# Road Quality and Age Prediction

This project aims to predict the quality of roads and estimate their age using three different machine learning models: Multi-Layer Perceptron (MLP), Gaussian Naive Bayes (GNB), and Bernoulli Naive Bayes (BNB). The project utilizes a dataset containing various features related to road conditions and employs these models to make predictions based on the input data.

## Dataset
The dataset used in this project is `rmdataset.csv`, which contains the following features:

- **IDMachines**
- **PeopleAtwork**
- **StreetLights**
- **Accidents**
- **DamagedMovers**
- **StRoadLength**
- **RoadCurvature**
- **HPBends**
- **RoadType**
- **RoadWidth**
- **AvgSpeed**
- **AgeOfRoad**
- **RoadSurface** (target variable)

## Models

### Multi-Layer Perceptron (MLP):
- Implemented in `MLP_nn.py`
- Trains a neural network to predict road quality and age.
- Saves the trained model to `mlp_model.h5`.

### Gaussian Naive Bayes (GNB):
- Implemented in `gnb1.py`
- Uses Gaussian Naive Bayes for classification.
- Evaluates the model using cross-validation and makes predictions on test data.

### Bernoulli Naive Bayes (BNB):
- Implemented in `bnb1.py`
- Uses Bernoulli Naive Bayes for classification.
- Evaluates the model using cross-validation and makes predictions on test data.

## Usage

### Data Preprocessing:
- Run `data.py` to preprocess the dataset and generate the `AgeOfRoad` feature.

### Training Models:
- Run `MLP_nn.py` to train the MLP model.
- Run `gnb1.py` to train and evaluate the GNB model.
- Run `bnb1.py` to train and evaluate the BNB model.

### Predicting Road Quality and Age:
- Use `predict_lifespan.py` to predict the age at which the road quality turns to 'Poor' using the trained MLP model.

## Requirements
Install the required dependencies using the following command:

```
pip install -r requirements.txt
```

## Description
This project leverages machine learning techniques to predict the quality and estimated age of roads based on various features. The dataset includes information such as the number of machines, people at work, streetlights, accidents, damaged movers, road length, curvature, bends, type, width, average speed, and age of the road. The target variable is the road surface condition, categorized as **'Poor', 'Avg', or 'Good'**.

The project implements three models:

- **MLP**: A neural network model that predicts road quality and age.
- **GNB**: A Gaussian Naive Bayes classifier that predicts road quality.
- **BNB**: A Bernoulli Naive Bayes classifier that predicts road quality.

The models are trained and evaluated using cross-validation, and their performance is measured using accuracy scores. The project also includes functionality to predict the age at which the road quality turns to 'Poor' using the trained MLP model.

For more details, refer to the documentation provided in the `Docs` folder.
