# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 17:52:26 2025

@author: sqril
"""
'''KNN Model'''

from sklearn.neighbors import KNeighborsClassifier

def KNN(df_train, df_test):
    X_train = df_train.drop(['label'], axis=1)
    y_train = df_train['label']
    X_test = df_test.drop(['label'],axis=1)
    y_test = df_test['label']
    
    #Initializing and Training Model
    knn = KNeighborsClassifier()
    knn.fit(X_train,y_train)

    #Making predictions
    y_pred_knn = knn.predict(X_test)
    
    return knn, X_test, y_test, y_pred_knn