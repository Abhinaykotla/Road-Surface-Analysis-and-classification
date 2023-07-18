from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn import *
import numpy as np
import pandas as pd
from sklearn import *
from sklearn.metrics import accuracy_score
df = pd.read_csv('rmdataset.csv')
df["RoadSurface"] = df["RoadSurface"].map({'Poor':0 ,'Avg':1 ,'Good':2})
data = df[["IDMachines","PeopleAtwork","StreetLights","Accidents","DamagedMovers","StRoadLength","RoadCurvature","HPBends","RoadType","RoadWidth","AvgSpeed","RoadSurface"]].to_numpy()
inputs = data[:,:-1]
outputs = data[:, -1]
training_inputs = inputs[:2500]
training_outputs = outputs[:2500]
testing_inputs = inputs[2500:]
testing_outputs = outputs[2500:]
classifier = GaussianNB()
classifier.fit(training_inputs, training_outputs)
predictions = classifier.predict(testing_inputs)
accuracy = 100.0 * accuracy_score(testing_outputs, predictions)
print("The accuracy of GNB Classifier on testing data is: " + str(accuracy))
testSet = [[1,8,544,41,1,1158,235.46,44,2,28,3.47]]
test = pd.DataFrame(testSet)
predictions = classifier.predict(test)
print('GNB percentage prediction on the first test set is:',predictions)
testSet = [[7,17,819,13,9,6165,19.62,0,0,95,0.28]]
test = pd.DataFrame(testSet)
predictions = classifier.predict(test)
print('GNB percentage prediction on the second test set is:',predictions)
testSet = [[5,15,739,33,6,2681,124.86,29,1,55,0.72]]
test = pd.DataFrame(testSet)
predictions = classifier.predict(test)
print('GNB percentage prediction on the third test set is:',predictions)