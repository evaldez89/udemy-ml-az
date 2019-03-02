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


#%% Fitting simple linear regression in the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#%% Predicting the test set result