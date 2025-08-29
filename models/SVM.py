# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 17:50:36 2025

@author: sqril
"""
'''SVM Model'''

from sklearn.svm import SVC

def SVM(df_train, df_test): 
    X_train = df_train.drop(['label'], axis=1)
    y_train = df_train['label']
    X_test = df_test.drop(['label'],axis=1)
    y_test = df_test['label']
    
    #Initializing and training model
    svm_model = SVC(kernel='linear', probability=True)  
    svm_model.fit(X_train, y_train)
    
    #Predicting Test Values
    y_pred_svm = svm_model.predict(X_test)
    
    return svm_model, X_test, y_test, y_pred_svm