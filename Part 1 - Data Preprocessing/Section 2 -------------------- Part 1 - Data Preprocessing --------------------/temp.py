#Data Processing

#Libraries

#numpy contains math stuff
import numpy as np

#basically builds/plots charts
import matplotlib.pyplot as plt

#manages datasets
import pandas as pd

#missing data...tool
from sklearn.preprocessing import Imputer

#IMPORTING DATASETS
#tells python to use Data.csv
dataset = pd.read_csv("Data.csv")

#for the bracketed stuff, look at it as x and y coords

#x = first three columns of Data.csv
x = dataset.iloc[:,:-1].values

#y=last column
y = dataset.iloc[:,3].values

#HOW TO HANDLE MISSING DATA FROM DATASETS
#looking for string in Data.csv "NaN", check documentation for strategy and axis values
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis=0)

imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

#print(x)

#ENCODING CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x = LabelEncoder()

#THE LINE BELOW IS BASICALLY GIVING ALL THE ELEMENTS IN COLUMN 0 AN ID, I.E. FRANCE = 0, SPAIN = 1, GERMANY = 2
#ISSUE THAT WILL ARISE, FRANCE WILL BE THE GREATEST PRIORITY FOR PROGRAMS, AND SPAIN MORE THAN GERMANY, ETC.
x[:,0] = labelencoder_x.fit_transform(x[:,0])

#BASICALLY SAYING COLUMN 0 IS CATEGORIES, DO NOT TREAT AS PRIORITIES
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)