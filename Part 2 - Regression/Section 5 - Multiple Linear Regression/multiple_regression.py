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

y_pred = regressor.predict(x_test)


#CODE BELOW IS STEP 2 OF BACKWARD ELMINATION
import statsmodels.formula.api as sm
x = np.append(arr = np.ones((50,1)). astype(int), values = x, axis = 1)
x_opt = x[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()

#STEP 3 AND 4 AND 5 IS BELOW

"""this is basically going to give you a table with every column and a bunch of different values.
P>|t| is where you wanna check how far above P value is above significance level.
remove the highest one, if any above your SL
in this case were removing index 2 of x data """
regressor_OLS.summary()

"""REMOVING INDEX 2"""
x_opt = x[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

"""REMOVING MORE INDEXES"""
x_opt = x[:,[0,3]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()





