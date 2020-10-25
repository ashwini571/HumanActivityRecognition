# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 20:09:25 2020

@author: Ashwini Ojha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#1 Reading Dataset
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

X_train = df_train.iloc[:,:-1]
y_train = df_train.iloc[:,:-1]

X_test = df_test.iloc[:,:-1]
y_test = df_test.iloc[:,:-1]
# Removing subjects col
X_train.drop(['subject'], axis=1, inplace =True)
X_test.drop(['subject'], axis=1, inplace = True)


#2 Data Pre-processing
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



#3 Dataset Analysis
x = ['Nuclear', 'Hydro', 'Gas', 'Oil', 'Coal', 'Biofuel']
energy = [5, 6, 15, 22, 24, 8]

x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, energy, color='green')
plt.xlabel("Energy Source")
plt.ylabel("Energy Output (GJ)")
plt.title("Energy output from various fuel sources")

plt.xticks(x_pos, x)

plt.show()


