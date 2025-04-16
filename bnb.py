from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
import os

def train_bnb_model():
    output = ""
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
        classifier = BernoulliNB()
        classifier.fit(X_train, y_train)

        # Evaluate the model using cross-validation
        cv_scores = cross_val_score(classifier, X_train, y_train, cv=5)
        output += f"Cross-validation accuracy scores: {cv_scores}\n"
        output += f"Mean cross-validation accuracy: {cv_scores.mean()}\n"

        # Make predictions and evaluate the model on the test set
        predictions = classifier.predict(X_test)
        accuracy = 100.0 * accuracy_score(y_test, predictions)
        output += f"The accuracy of BNB Classifier on testing data is: {accuracy}\n"

        # Ensure the directory exists
        os.makedirs('models/bnb', exist_ok=True)

        # Save the model and scaler
        joblib.dump(classifier, 'models/bnb/bnb_model.pkl')
        np.save('models/bnb/bnb_scaler.npy', scaler.mean_)
        np.save('models/bnb/bnb_scaler_scale.npy', scaler.scale_)
        output += "Model and scaler saved\n"

        return output
    except Exception as e:
        return f"An error occurred during training: {e}"

def test_bnb_model(test_sets):
    output = ""
    test_original=test_sets.copy()
    try:
        # Load the saved model and scaler
        classifier = joblib.load('models/bnb/bnb_model.pkl')
        scaler = StandardScaler()
        scaler.mean_ = np.load('models/bnb/bnb_scaler.npy')
        scaler.scale_ = np.load('models/bnb/bnb_scaler_scale.npy')
    except FileNotFoundError:
        return "Model or scaler file not found"
    except Exception as e:
        return f"An error occurred while loading the model or scaler: {e}"

    try:
        # Scale the input data
        test_sets = scaler.transform(test_sets)

        # Make predictions
        predictions = classifier.predict(test_sets)
        output_array = []
        output += "Test Cases and Predictions:\n"
        output += "-" * 50 + "\n"
        
        features = ["IDMachines", "PeopleAtwork", "StreetLights", "Accidents", 
                   "DamagedMovers", "StRoadLength", "RoadCurvature", "HPBends", 
                   "RoadType", "AvgSpeed", "RoadWidth", "AgeOfRoad"]
        
        for i in range(len(predictions)):
            if predictions[i] == 0:
                prediction = "Poor quality"
            elif predictions[i] == 1:
                prediction = "Avg quality"
            else:
                prediction = "Good quality"
            output_array.append(prediction)
            
            # Add test case details
            output += f"\nTest Case {i+1}:\n"
            for j, feature in enumerate(features):
                output += f"{feature}: {test_original[i][j]:.2f}\n"
            output += f"Prediction: {prediction}\n"
            output += "-" * 50 + "\n"
            
        output += f'\nSummary of predictions:\n{output_array}\n'
        return output
    except Exception as e:
        return f"An error occurred during prediction: {e}"

if __name__ == "__main__":
    result = train_bnb_model()
    if isinstance(result, str):
        print(result)
    else:
        print(result)

        # Example test sets
        test_sets = [
            [1, 8, 544, 41, 1, 1158, 235.46, 44, 2, 28, 3.47, 2],
            [7, 17, 819, 13, 9, 6165, 19.62, 0, 0, 95, 0.28, 25],
            [5, 15, 739, 33, 6, 2681, 124.86, 29, 1, 55, 0.72, 5]
        ]

        result = test_bnb_model(test_sets)
        if isinstance(result, str):
            print(result)
        else:
            print(result)