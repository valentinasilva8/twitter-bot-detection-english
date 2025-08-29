# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 17:37:11 2025

"""
'''Preprocessing Data'''

from sklearn.preprocessing import RobustScaler, MinMaxScaler

def PreprocessData(df):
    df = df.dropna() #drop null values
    df = df.select_dtypes(include=['float64', 'int64']) #dropping nonnumeric columns columns
    
    #dropping columns already scaled or categorical
    exclude_columns = ['label', 'is_verified']
    columns_to_scale = [col for col in df.columns if col not in exclude_columns]
    
    #applying robust scaler
    robust_scaler = RobustScaler()
    df[columns_to_scale] = robust_scaler.fit_transform(df[columns_to_scale])
    
    #applying minmax scaler to bring the range to [0, 1]
    exclude_columns1 = ['label', 'is_verified']
    columns_to_scale1 = [col for col in df.columns if col not in exclude_columns1]
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    df[columns_to_scale1] = min_max_scaler.fit_transform(df[columns_to_scale1])
    
    return df