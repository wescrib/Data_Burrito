# -*- coding: utf-8 -*-
"""
Created on Wed May  2 21:40:09 2018

@author: scrib
"""

#Data Processing

#Libraries

#numpy contains math stuff
import numpy as np

#manages datasets
import pandas as pd

import matplotlib.pyplot as plt


#IMPORTING DATASETS
#tells python to use Data.csv
dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state = 0)


#FEATURE SCALING
"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""

