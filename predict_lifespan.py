import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_prepare_model():

    # Load the saved model
    model = load_model('models/mlp_model.h5')

    # Define the features used in the model
    features = ["IDMachines", "PeopleAtwork", "StreetLights", "Accidents", "DamagedMovers", "StRoadLength", "RoadCurvature", "HPBends", "RoadType", "RoadWidth", "AvgSpeed", "AgeOfRoad"]

    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Fit the scaler on the training data
    # Load dataset
    df = pd.read_csv('rmdataset.csv')

    # Data preprocessing
    df["RoadSurface"] = df["RoadSurface"].map({'Poor': 0, 'Avg': 1, 'Good': 2})

    # Feature selection
    X = df[features]
    y = df["RoadSurface"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    # Feature scaling
    scaler.fit(X_train)

    return model, scaler

def predict_poor_quality_age(inputs):
    """
    Predict the age at which the road quality turns to 'Poor'.
    
    Parameters:
    inputs (list): A list of 11 input features excluding age.
    
    Returns:
    int: The age at which the road quality turns to 'Poor'.
    """
    # Load and prepare the model
    model, scaler = load_and_prepare_model()

    for age in range(1, 31):
        # Create the input array with the given inputs and the current age
        test_set = inputs + [age]
        
        # Scale the input
        test_set_scaled = scaler.transform([test_set])
        
        # Predict the road quality
        prediction = model.predict(test_set_scaled)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        # Check if the predicted class is 'Poor' (0)
        if predicted_class == 0:
            return age
    
    return -1

# # Test the function
# if __name__ == "__main__":
#     # Example usage
#     inputs = [7, 17, 819, 13, 9, 6165, 19.62, 0, 0, 200, 0.28]
#     poor_quality_age = predict_poor_quality_age(inputs)
#     print(f"The road quality turns to 'Poor' at age of: {poor_quality_age - 1} to {poor_quality_age} years")