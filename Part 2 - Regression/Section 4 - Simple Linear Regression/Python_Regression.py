# -*- coding: utf-8 -*-
"""
Created on Tue May  1 22:08:49 2018

@author: scrib
"""
#IMPORTING LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING DATASET
data = pd.read_csv('Salary_Data.csv')
x = data.iloc[:,:-1].values
y = data.iloc[:,1].values

#SPLITTING DATA INTO TEST AND TRAINING SETS
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 1/3, random_state = 0)

#SIMPLE LINEAR REGRESSION TO TRAINING DATASET
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train,)

"""regressor is the machine in this case that is learning. this is a super basic machin learning model."""
