import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

def train_nn():
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

        # Convert labels to categorical one-hot encoding
        y = to_categorical(y, num_classes=3)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

        # Feature scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Build the neural network model
        model = Sequential()
        model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(3, activation='softmax'))

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=15, batch_size=10, validation_split=0.2)

        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Model accuracy on test data: {accuracy * 100:.2f}%")

        # Save the model and scaler
        model.save('models/mlp_model.h5')
        np.save('models/scaler.npy', scaler.mean_)
        np.save('models/scaler_scale.npy', scaler.scale_)
        print("Model and scaler saved")

        return model, scaler
    except Exception as e:
        return f"An error occurred during training: {e}"

def test_nn(test_sets):
    try:
        # Load the saved model and scaler
        model = load_model('models/mlp_model.h5')
        scaler = StandardScaler()
        scaler.mean_ = np.load('models/scaler.npy')
        scaler.scale_ = np.load('models/scaler_scale.npy')
    except FileNotFoundError:
        return "Model or scaler file not found"
    except Exception as e:
        return f"An error occurred while loading the model or scaler: {e}"

    try:
        # Scale the input data
        test_sets = scaler.transform(test_sets)

        # Make predictions
        predictions = model.predict(test_sets)
        predicted_classes = np.argmax(predictions, axis=1)
        print(f'MLP predictions on the test sets: {predicted_classes}')
        return predicted_classes
    except Exception as e:
        return f"An error occurred during prediction: {e}"

if __name__ == "__main__":
    result = train_nn()
    if isinstance(result, str):
        print(result)
    else:
        model, scaler = result

        # Example test sets
        test_sets = [
            [1, 8, 544, 41, 1, 1158, 235.46, 44, 2, 28, 3.47, 2],
            [7, 17, 819, 13, 9, 6165, 19.62, 0, 0, 95, 0.28, 20],
            [5, 15, 739, 33, 6, 2681, 124.86, 29, 1, 55, 0.72, 5]
        ]

        result = test_nn(test_sets)
        if isinstance(result, str):
            print(result)