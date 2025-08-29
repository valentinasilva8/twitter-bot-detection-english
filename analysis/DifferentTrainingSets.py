# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 17:45:06 2025

"""
'''Creating different sets to train and test the models on'''

import SplittingDataset as SplittingDataset


def DifferentTrainingSets(df_train,df_test):
    #splitting dataset into train and test
    #df_train, df_test = SplittingDataset(df)
    
    #Top 5 Features Based on Correlation With Label
    corr_top5 = df_train.corr()['label'].abs().sort_values(ascending=False)
    
    #Selecting top 5 features
    top_5_features = corr_top5.head(6).index.tolist() #using 6 because label is included
    top_5_features
    
    #Dataframes for top 5 features
    df_train_top5 = df_train[top_5_features]
    df_test_top5 = df_test[top_5_features]
    
    #Top 10 Features Based on Correlation With Label
    corr_top10 = df_train.corr()['label'].abs().sort_values(ascending=False)
    
    #Selecting top 10 features
    top_10_features = corr_top10.head(11).index.tolist() #using 11 because label is included
    top_10_features
    
    #Dataframes for top 10 features
    df_train_top10 = df_train[top_10_features]
    df_test_top10 = df_test[top_10_features]
    
    return df_train_top5, df_test_top5, df_train_top10, df_test_top10, df_train, df_test