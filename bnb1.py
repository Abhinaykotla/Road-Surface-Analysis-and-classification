from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('rmdataset.csv')

# Data preprocessing
df["RoadSurface"] = df["RoadSurface"].map({'Poor': 0, 'Avg': 1, 'Good': 2})

# Feature selection
features = ["IDMachines", "PeopleAtwork", "StreetLights", "Accidents", "DamagedMovers", "StRoadLength", "RoadCurvature", "HPBends", "RoadType", "RoadWidth", "AvgSpeed", "AgeOfRoad"]
X = df[features]
y = df["RoadSurface"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the classifier
classifier = BernoulliNB()
classifier.fit(X_train, y_train)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(classifier, X_train, y_train, cv=5)
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean cross-validation accuracy: {cv_scores.mean()}")

# Make predictions and evaluate the model on the test set
predictions = classifier.predict(X_test)
accuracy = 100.0 * accuracy_score(y_test, predictions)
print("The accuracy of BNB Classifier on testing data is: " + str(accuracy))

# Test the model with new data
test_sets = [
        [1, 8, 544, 41, 1, 1158, 235.46, 44, 2, 28, 3.47, 2],
    [7, 17, 819, 13, 9, 6165, 19.62, 0, 0, 95, 0.28, 25],
    [5, 15, 739, 33, 6, 2681, 124.86, 29, 1, 55, 0.72, 5]
]

for i, test_set in enumerate(test_sets, start=1):
    test = pd.DataFrame([test_set], columns=features)
    test = scaler.transform(test)
    prediction = classifier.predict(test)
    print(f'BNBC prediction on the test set {i} is:', prediction)