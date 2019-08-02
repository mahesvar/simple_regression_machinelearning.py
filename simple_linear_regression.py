# this is a simple linear regression machine learning model
# which predicts salaries of the employees from the trained data set
# here LinearRegression library is used as the machine to train the data


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import Imputer

dataset = pd.read_csv("Salary_Data.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

# split the data \set into the training set and test set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 1/3,random_state = 0)
  
# fitting simple linear regression to the training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# predicting the text set results
y_pred = regressor.predict(x_test)

# visualising the training set

plt.scatter(x_train,y_train, color = "red")
plt.plot(x_train,regressor.predict(x_train), color = "blue")
plt.title('salary prediction(train set)')
plt.xlabel('years of experience')
plt.ylabel("salary")
plt.grid()
plt.show()

# visualising the test set

plt.scatter(x_test,y_test, color = "red")
#regressor is already trained on training set
plt.plot(x_train,regressor.predict(x_train), color = "blue")
plt.title('salary prediction(test set)')
plt.xlabel('years of experience')
plt.ylabel("salary")
plt.grid()
plt.show()
