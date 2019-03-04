# Data Preprocessing Template

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import statsmodels.formula.api as sm


# Change format 
np.set_printoptions(formatter=dict(float=lambda t: "%.0f" % t))

def get_data(dataset, x_index: int, y_index: int):
    x = dataset.iloc[:, :x_index].values
    y = dataset.iloc[:, y_index].values
    
    return x, y

def encode_categorical_data(data_set, col_index: int):
    labelencoder = LabelEncoder()
    data_set[:, col_index] = labelencoder.fit_transform(data_set[:, col_index])

    onehotencoder = OneHotEncoder(categorical_features=[col_index])
    return onehotencoder.fit_transform(data_set).toarray()

def plot_sets(x_set, y_set, x_pred):
    plt.scatter(x_set, y_set, color='red')
    plt.plot(X_train, x_pred, color='blue')
    plt.title('Salary vs Expirience (Training set)')
    plt.xlabel('Years of Expirience ')
    plt.ylabel('Salary ')
    plt.show()

# method for simple linear regression
def simple_linear_regressor(x_set, y_set, predict_y: bool = False):
    # Fitting simple linear regression in the training set
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test) if predict_y == True else None
    x_pred = regressor.predict(X_train)

    return x_pred, y_pred

#%% Importing the dataset
doc_path = '50_Startups.csv'
dataset = pd.read_csv(doc_path)

#%% Get X and y
X, y = get_data(dataset, -1, 4)

#%% Encode
X = encode_categorical_data(X, 3)

#%% Avoiding the Dummy Variable Trap
X = X[:, 1:]  # The library already does it

#%% Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

#%% Show plot for simple linear regression
# Set plot
x_pred, y_pred = simple_linear_regressor(X_train, y_train, True) # Train
x_pred, y_pred = simple_linear_regressor(X_test, y_test, True) # Test

#%%
# Show plot. --- Only for simple linear regression (for now)
plot_sets(X_train, y_train, x_pred) # Train
plot_sets(X_test, y_test, x_pred) # Test

#%% Backward Elimination
# Append b0 (array full of ones) cause the library statsmodels does not do it automatically
X = np.append(np.ones((50, 1)).astype(int), X, 1)

#%% Backward Elimination - Step 2, fit the full model with all possible predictors
# X_opt would be the optimal matrix
def backwardElimination(x, sl):  # P-values only
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

def backwardEliminationAdjusted(x, SL):  # P-values and Adjusted R Squared
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# X_Modeled = backwardElimination(X_opt, SL)
X_Modeled = backwardEliminationAdjusted(X_opt, SL)
