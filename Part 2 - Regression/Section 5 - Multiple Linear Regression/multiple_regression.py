# -*- coding: utf-8 -*-
"""
Created on Wed May  2 21:40:09 2018

@author: scrib

profit is dependant var
all other columns are independant
"""

#Libraries

#numpy contains math stuff
import numpy as np

#manages datasets
import pandas as pd

import matplotlib.pyplot as plt


#IMPORTING DATASETS
#tells python to use Data.csv
dataset = pd.read_csv("50_Startups.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()

""" using the third index for dummy vars"""
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

#manually avoid dummy var trap
""" removes first column """
x = x[:,1:]



from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state = 0)


#FEATURE SCALING
"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""

#fitting multiple linear regression to the rtainint set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

