# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


def get_data(doc_path: str, x_inverse_index: int, y_index: int):
    dataset = pd.read_csv('Data.csv')
    x = dataset.iloc[:, :x_inverse_index].values
    y = dataset.iloc[:, y_index].values
    
    return dataset, x, y


# Importing the dataset
dataset, X, y = get_data('Salary_Data.csv', -1, 3)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
