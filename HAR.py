# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 20:09:25 2020

@author: Ashwini Ojha
"""

import pandas as pd
import numpy as np

# Reading Dataset
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

X_train = df_train.iloc[:,:-1]
y_train = df_train.iloc[:,:-1]

X_test = df_test.iloc[:,:-1]
y_test = df_test.iloc[:,:-1]
# Removing subjects col
X_train.drop(['subject'], axis=1, inplace =True)
X_test.drop(['subject'], axis=1, inplace = True)


## Data Pre-processing
# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer.fit_transform(X_train)
imputer.transform(X_test)

#Removing unnecessary char
cols = X_train.columns
cols = cols.str.replace("[()]", '') 
cols = cols.str.replace("-", '')
cols = cols.str.replace(",", '')

X_train.columns = cols
X_test.columns = cols


