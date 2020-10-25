# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 20:09:25 2020

@author: Ashwini Ojha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#1 Reading Dataset
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

X_train = df_train.iloc[:,:-1]
y_train = df_train['Activity']

X_test = df_test.iloc[:,:-1]
y_test = df_test['Activity']
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
# Count vs activity
bars = y_train.value_counts(sort=False)

fig, ax = plt.subplots(figsize =(16, 9)) 
ax.barh(bars.index, bars)
# Add annotation to bars 
for i in ax.patches: 
    plt.text(i.get_width()+0.2, i.get_y()+0.5,  
             str(round((i.get_width()), 2)), 
             fontsize = 10, fontweight ='bold', 
             color ='grey') 
plt.xlabel("Count")
plt.ylabel("Activity")
plt.title("Count vs Activity")
plt.show()


# tBodyAccMagmean vs Activities
sns.boxplot(x = y_train, y = "tBodyAccMagmean", showfliers = False, data = X_train)
plt.axhline(y = -0.65, linestyle = "--")
plt.axhline(y = 0, linestyle = "--")
plt.title("Box plot of tBodyAccMagmean", fontsize = 15)
plt.ylabel("Accelerator Body Mean", fontsize = 15)
plt.xlabel("Activity Name", fontsize = 15)
plt.tick_params(labelsize = 15)
plt.xticks(rotation = 40)
plt.show()



