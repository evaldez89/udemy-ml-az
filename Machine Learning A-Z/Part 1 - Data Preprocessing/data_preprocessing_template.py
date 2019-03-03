# Data Preprocessing Template

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def get_data(dataset, x_index: int, y_index: int):
    x = dataset.iloc[:, :x_index].values
    y = dataset.iloc[:, y_index].values
    
    return x, y

#%% Importing the dataset
doc_path = 'Salary_Data.csv'
dataset = pd.read_csv(doc_path)

#%% Get X and y
X, y = get_data(dataset, -1, 1)

#%% Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state=0)

# method for simple linear regression
def simple_linear_regressor(x_set, y_set):
    # Fitting simple linear regression in the training set
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    # y_pred = regressor.predict(X_test)  # No needed just to test
    x_pred = regressor.predict(X_train)

    plt.scatter(x_set, y_set, color='red')
    plt.plot(X_train, x_pred, color='blue')
    plt.title('Salary vs Expirience (Training set)')
    plt.xlabel('Years of Expirience ')
    plt.ylabel('Salary ')
    plt.show()

#%% Show plot
simple_linear_regressor(X_train, y_train)
simple_linear_regressor(X_test, y_test)
