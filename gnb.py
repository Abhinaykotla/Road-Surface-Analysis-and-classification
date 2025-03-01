from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
import os

def train_gnb_model():
    try:
        # Load dataset
        df = pd.read_csv('datasets/rmdataset.csv')
    except FileNotFoundError:
        return "File not found"
    except Exception as e:
        return f"An error occurred while loading the dataset: {e}"

    try:
        # Data preprocessing
        df["RoadSurface"] = df["RoadSurface"].map({'Poor': 0, 'Avg': 1, 'Good': 2})

        # Feature selection
        features = ["IDMachines", "PeopleAtwork", "StreetLights", "Accidents", "DamagedMovers", "StRoadLength", "RoadCurvature", "HPBends", "RoadType", "RoadWidth", "AvgSpeed", "AgeOfRoad"]
        X = df[features]
        y = df["RoadSurface"]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

        # Feature scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Initialize and train the classifier
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)

        # Evaluate the model using cross-validation
        cv_scores = cross_val_score(classifier, X_train, y_train, cv=5)
        print(f"Cross-validation accuracy scores: {cv_scores}")
        print(f"Mean cross-validation accuracy: {cv_scores.mean()}")

        # Make predictions and evaluate the model on the test set
        predictions = classifier.predict(X_test)
        accuracy = 100.0 * accuracy_score(y_test, predictions)
        print("The accuracy of GNB Classifier on testing data is: " + str(accuracy))

        # Ensure the directory exists
        os.makedirs('models/gnb', exist_ok=True)

        # Save the model and scaler
        joblib.dump(classifier, 'models/gnb/gnb_model.pkl')
        np.save('models/gnb/gnb_scaler.npy', scaler.mean_)
        np.save('models/gnb/gnb_scaler_scale.npy', scaler.scale_)
        print("Model and scaler saved")

        return classifier, scaler, features
    except Exception as e:
        return f"An error occurred during training: {e}"

def test_gnb_model(test_sets):
    try:
        # Load the saved model and scaler
        classifier = joblib.load('models/gnb/gnb_model.pkl')
        scaler = StandardScaler()
        scaler.mean_ = np.load('models/gnb/gnb_scaler.npy')
        scaler.scale_ = np.load('models/gnb/gnb_scaler_scale.npy')
    except FileNotFoundError:
        return "Model or scaler file not found"
    except Exception as e:
        return f"An error occurred while loading the model or scaler: {e}"

    try:
        # Scale the input data
        test_sets = scaler.transform(test_sets)

        # Make predictions
        predictions = classifier.predict(test_sets)
        print(f'GNB predictions on the test sets: {predictions}')
        return predictions
    except Exception as e:
        return f"An error occurred during prediction: {e}"

if __name__ == "__main__":
    result = train_gnb_model()
    if isinstance(result, str):
        print(result)
    else:
        classifier, scaler, features = result

        # Example test sets
        test_sets = [
            [1, 8, 544, 41, 1, 1158, 235.46, 44, 2, 28, 3.47, 2],
            [7, 17, 819, 13, 9, 6165, 19.62, 0, 0, 95, 0.28, 25],
            [5, 15, 739, 33, 6, 2681, 124.86, 29, 1, 55, 0.72, 5]
        ]

        result = test_gnb_model(test_sets)
        if isinstance(result, str):
            print(result)
        else:
            print(f"Predicted classes: {result}")