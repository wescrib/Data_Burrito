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

"""regressor IS the machine that is learning. this is a super basic machin learning model."""
"""training data is basically the data that the machine will learn from. the test data is where the learning will be applied, i.e. THE PREDICTION. BAM!"""

#WILL PREDICT TEST SET RESULTS
y_predict = regressor.predict(x_test)
"""after this line, compare y_test data and the new y_predict data | y_test is legit real data"""

#VISUALIZATION OF TRAINING DATA
"""matplotlib is finally making its appearance"""
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train,regressor.predict(x_train), color='green')
plt.title("SALARY vs EXPERIENCE (Training Data)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#VISUALIZATION OF TEST DATA
"""matplotlib is finally making its appearance"""
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train,regressor.predict(x_train), color='green')
plt.title("SALARY vs EXPERIENCE (Training Data)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()