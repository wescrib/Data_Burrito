"""
this dude is applying for a new job, and told HR that his salary at his current salary is 160k. HR found this oddly high.
we need to use some kind of regression algorithm to figure out if he is being honest or not
"""


# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
#dont need to split because theres so little data as it is
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#build linear regression dataset to compare to the poly regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#build poly regression dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)

""" fitting poly fit into a linear regression model """
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

#visual of linear regression

"""does the plotting, literally x and y coords"""
plt.scatter(x, y, color = 'red')
""" the prediction, the slope """
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


#visual of poly regression
"""does the plotting, literally x and y coords"""
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x, y, color = 'red')
""" the prediction, the slope
use lin_reg2 because thats using your x_poly data, then predict the x_poly data, not just x """
plt.plot(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)), color='blue')
plt.title("Truth or Bluff (Poly Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


#predicting future results with linear reg
""" looking for where the salary will when x = 6.5, still a garbage prediction cause this is a regular linear model ..and the data is not linear"""
lin_reg.predict(6.5)

""" this outputs 330k, not accurate at all"""



#predicting future results with poly reg
lin_reg2.predict(poly_reg.fit_transform(6.5))

""" this outputs about 158 - 159k """





