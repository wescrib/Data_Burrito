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

print(x)