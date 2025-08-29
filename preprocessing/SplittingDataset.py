# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 17:41:35 2025

@author: sqril
"""
'''Splitting Dataset to ensure all models use the same data'''

import pandas as pd
from sklearn.model_selection import train_test_split

def SplittingDataset(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['label'],axis=1), df['label'], test_size=0.2, random_state=42)
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    
    return df_train, df_test