#loading libraries
import pandas as pd
import numpy as np

#loading dataset
dataset = pd.read_csv('training_data.csv')
X_train = dataset.iloc[:, [4,5,6,7,8,13,15,16,10]].values
y_train = dataset.iloc[:, 17].values



## Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


pred = pd.read_csv('testing_data.csv')
X_pred = pred.iloc[:, [4,5,6,7,8,13,15,16,10]].values

#categorical data 
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X_train[:, 4] = labelencoder_X.fit_transform(X_train[:, 4])
X_train[:, 6] = labelencoder_X.fit_transform(X_train[:, 6])
X_pred[:, 4] = labelencoder_X.fit_transform(X_pred[:, 4])
X_pred[:, 6] = labelencoder_X.fit_transform(X_pred[:, 6])

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)

#Time Conversion
for i in range (0,52676):
    time = X_train[i, 5]
    min = time[0:2]
    min = float(min) * 60
    sec = time[3:7]
    sec = float(sec) + min
    X_train[i, 5] = sec
    
    t = X_train[i,8].split("+")
    t = t[0]+t[1]
    X_train[i,8] = t
    
    
for i in range (0,28365):
    time = X_pred[i, 5]
    min = time[0:2]
    min = float(min) * 60
    sec = time[3:7]
    sec = float(sec) + min
    X_pred[i, 5] = sec
    
    t = X_pred[i,8].split("+")
    t = t[0]+t[1]
    X_pred[i,8] = t

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_pred = sc_X.transform(X_pred)


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_pred)


## Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm4 = confusion_matrix(y_test, y_pred)

 
p_pred = [None]*28365

for i in range(0,28365):
    k = y_pred[i]
    if k == 0: 
        p_pred[i] = 'Black checkmated'
    if k == 1: 
        p_pred[i] = 'Black forfeits by disconnection'
    if k == 2: 
        p_pred[i] = 'Black forfeits on time'
    if k == 3: 
        p_pred[i] = 'Black ran out of time and White has no material to mate'
    if k == 4: 
        p_pred[i] = 'Black resigns'
    if k == 5: 
        p_pred[i] = 'Game drawn by mutual agreement'
    if k == 6: 
        p_pred[i] = 'Game drawn by repetition'
    if k == 7: 
        p_pred[i] = 'Game drawn by stalemate'
    if k == 8: 
        p_pred[i] = 'Game drawn by the 50 move rule'
    if k == 9: 
        p_pred[i] = 'Neither player has mating material'
    if k == 10: 
        p_pred[i] = 'White checkmated'
    if k == 11: 
        p_pred[i] = 'White forfeits by disconnection'
    if k == 12: 
        p_pred[i] = 'White forfeits on time'
    if k == 13: 
        p_pred[i] = 'White ran out of time and Black has no material to mate'
    if k == 14: 
        p_pred[i] = 'White resigns'

id = pred.iloc[:, 0].values
df = pd.DataFrame({'commentaries': p_pred, 'id': id})
