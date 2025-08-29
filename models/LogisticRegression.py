# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 17:46:59 2025

@author: sqril
"""
'''Logistic Regression Model'''

from sklearn.linear_model import LogisticRegression as LR

def LogisticRegression(df_train, df_test):
    X_train = df_train.drop(['label'], axis=1)
    y_train = df_train['label']
    X_test = df_test.drop(['label'],axis=1)
    y_test = df_test['label']
    
    #Initializing and training model
    lr_model = LR()
    lr_model.fit(X_train, y_train)
    
    #Predicting values
    y_pred_lr = lr_model.predict(X_test)
    
    return lr_model, X_test, y_test, y_pred_lr