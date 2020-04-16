import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

import re 

dataset = pd.read_csv('/Users/ftaga/desktop/thesispython/winequality-red.csv')
print(dataset.shape)
print(dataset.describe())

dataset.isnull().any()
dataset = dataset.fillna(method='ffill')
 
categories = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']
X = dataset[categories].values
y = dataset['quality'].values

regressor = LinearRegression().fit(X,y) 
R2_true = regressor.score(X, y)
print(R2_true)

# for i in range(0,1598):    
#     regressor_i = LinearRegression.fit(X[i,], y[i,])
#     R2_aux[i] = regressor_i.score(X[i,], y[i,])


print(X.shape, y.shape,X, y)