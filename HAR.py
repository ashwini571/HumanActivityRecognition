# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 20:09:25 2020

@author: Ashwini Ojha
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.svm import LinearSVC


#-----------------------------------Machine Learning Approach----------------------------------------
 
#1 Reading Dataset
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
activities = list(df_train['Activity'].unique())

encoder = LabelEncoder()
X_train = df_train.iloc[:,:-1]
y_train = df_train['Activity']
y_train = pd.DataFrame(encoder.fit_transform(y_train))

X_test = df_test.iloc[:,:-1]
y_test = df_test['Activity']
y_test = pd.DataFrame(encoder.fit_transform(y_test))

# Removing subjects col
X_train.drop(['subject'], axis=1, inplace =True)
X_test.drop(['subject'], axis=1, inplace = True)


#2 Data Pre-processing
# Taking care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
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
def plot_activity_count():
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
def plot_activitiesVsBodyAccMean():
    sns.boxplot(x = y_train, y = "tBodyAccMagmean", showfliers = False, data = X_train)
    plt.axhline(y = -0.65, linestyle = "--")
    plt.axhline(y = 0, linestyle = "--")
    plt.title("Box plot of tBodyAccMagmean", fontsize = 15)
    plt.ylabel("Accelerator Body Mean", fontsize = 15)
    plt.xlabel("Activity Name", fontsize = 15)
    plt.tick_params(labelsize = 15)
    plt.xticks(rotation = 40)
    plt.show()

def plot_tSNE(p,l):
    data = X_train
    tsne = TSNE(n_components=2, perplexity = p, learning_rate=l)
    dim_reduced = tsne.fit_transform(data)
    
    df = pd.DataFrame(data = {'Dim-1':tsne.embedding_[:,0],'Dim-2':tsne.embedding_[:,1],'activities':y_train})
    sns.set_style('whitegrid')
    sns.scatterplot(x='Dim-1', y='Dim-2', data=df,hue='activities' )
    plt.title("TSNE Plot, Perplexity: "+str(p))
    plt.show()

#4 ML Models
def plot_cf(clf,predicted):
    plot_confusion_matrix(clf,X_test, y_test,display_labels = ["Laying","Sitting","Standing","Walking","Walking_Downstairs","Walking_Upstairs"])  
    plt.show() 
    
def print_performance(clf):
    predicted = clf.predict(X_test)
    train_score = clf.score(X_train,y_train)
    test_score = clf.score(X_test,y_test)
    # Accuracy
    print('---------------------------------')
    print('|      Accuracy (Train Set)     |')
    print('---------------------------------')
    print(str(train_score)+"%\n")
    print('--------------------------------')
    print('|      Accuracy (Test Set)     |')
    print('--------------------------------')
    print(str(test_score)+"%\n")    
    
    # Precision, Recall, F1-score
    precision, recall, fscore, support = score(y_test, predicted,average='macro')
    print('---------------------')
    print('|      Precision     |')
    print('---------------------')
    print(str(precision)+"%\n")
    print('---------------------')
    print('|      Recall     |')
    print('---------------------')
    print(str(recall)+"%\n")
    print('---------------------')
    print('|      F1-Score     |')
    print('---------------------')
    print(str(fscore)+"%\n")
    
    # Confusion Matrix
    plot_cf(clf,predicted)

# Logistic Regression
    
clf_lr = LogisticRegression(multi_class='ovr', max_iter=1000)
clf_lr.fit(X_train,y_train.values.flatten())
print_performance(clf_lr)

# RBF SVM
clf_rbf = SVC()
clf_rbf.fit(X_train, y_train.values.flatten())
print_performance(clf_rbf)

#Linear SVM
clf_linear_svm = LinearSVC()
clf_linear_svm.fit(X_train, y_train.values.flatten())
print_performance(clf_linear_svm)




#----------------------------------------------Deep Learning approach---------------------------------------

