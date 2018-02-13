#loading libraries
import pandas as pd
import numpy as np


#loading dataset
dataset = pd.read_csv('training_data.csv')
X = dataset.iloc[:, [4,5,6,7,8,13,15,16]].values
y = dataset.iloc[:, 17].values

pred = pd.read_csv('testing_data.csv')
X_pred = pred.iloc[:, [4,5,6,7,8,13,15,16]].values


#categorical data 
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])
X[:, 6] = labelencoder_X.fit_transform(X[:, 6])

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Time Conversion
for i in range (0,52676):
    time = X[i, 5]
    min = time[0:2]
    min = float(min) * 60
    sec = time[3:7]
    sec = float(sec) + min
    X[i, 5] = sec
    

#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)


#applying lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)


#Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

## Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm4 = confusion_matrix(y_test, y_pred)
