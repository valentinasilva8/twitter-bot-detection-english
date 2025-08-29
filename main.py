# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 17:55:27 2025

"""

#0 is human 1 is bot

import pandas as pd
from analysis.Info import Info 
from analysis.StatAnalysis import StatAnalysis as Stats
from analysis.OutlierAnalysis import OutlierAnalysis
from preprocessing.PreprocessData import PreprocessData
from preprocessing.SplittingDataset import SplittingDataset
from analysis.ExploratoryAnalysis import ExploratoryAnalysis as Explore
from analysis.DifferentTrainingSets import DifferentTrainingSets
from models.LogisticRegression import LogisticRegression
from models.RandomForest import RandomForest
from models.SVM import SVM
from models.KNN import KNN
from models.LSTM import train_hybrid_lstm_model, train_bd_lstm_model
from analysis.EvaluateModel import EvaluateModel as Evaluate
from analysis.Kmeans_pca import Kmeans_pca

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#importing dataset
df = pd.read_csv("data/twitter_english_data.csv")
df.columns
#df = df.drop(['is_verified'], axis =1) testing accuracy without is_verified column

#getting information about dataset
Info(df)

#getting statistical values about the dataset
Stats(df)

#doing outlier analysis
OutlierAnalysis(df)
#did not remove outliers because there were a great amount which indicate these are likely true datapoints


#Preprocessing data - scaling and removing non-numeric values
df_scaled = PreprocessData(df)
df_scaled.head()


#Splitting the dataset into training and testing
df_train, df_test = SplittingDataset(df_scaled)


#Doing Exploratory Analysis on Training dataset for feature selection
Explore(df_train)

#Creating different training sets for the models
df_train_top5, df_test_top5, df_train_top10, df_test_top10, df_train, df_test = DifferentTrainingSets(df_train, df_test)

df_train_top5.columns
df_train_top10.columns

'''Classification'''
#Model Initialization and Training
#Logistic Regression Models using different datasets
model_lr5, X_test_lr5, y_test_lr5, y_pred_lr5 = LogisticRegression(df_train_top5, df_test_top5)
model_lr10, X_test_lr10, y_test_lr10, y_pred_lr10 = LogisticRegression(df_train_top10, df_test_top10)
model_lr, X_test_lr, y_test_lr, y_pred_lr = LogisticRegression(df_train, df_test)

#Random Forrest Models using different datasets  
model_rf5, X_test_rf5, y_test_rf5, y_pred_rf5 = RandomForest(df_train_top5, df_test_top5)
model_rf10, X_test_rf10, y_test_rf10, y_pred_rf10 = RandomForest(df_train_top10, df_test_top10)
model_rf, X_test_rf, y_test_rf, y_pred_rf = RandomForest(df_train, df_test) 

#SVM Models using different datasets
model_svm5, X_test_svm5, y_test_svm5, y_pred_svm5 = SVM(df_train_top5, df_test_top5)
model_svm10, X_test_svm10, y_test_svm10, y_pred_svm10 = SVM(df_train_top10, df_test_top10)
model_svm, X_test_svm, y_test_svm, y_pred_svm = SVM(df_train, df_test)     
 
#KNN Models using different datasets
model_knn5, X_test_knn5, y_test_knn5, y_pred_knn5 = KNN(df_train_top5, df_test_top5)
model_knn10, X_test_knn10, y_test_knn10, y_pred_knn10 = KNN(df_train_top10, df_test_top10)
model_knn, X_test_knn, y_test_knn, y_pred_knn = KNN(df_train, df_test) 
 

#Evaluating Models
Evaluate(model_lr5, X_test_lr5, y_test_lr5, y_pred_lr5)
Evaluate(model_rf5, X_test_rf5, y_test_rf5, y_pred_rf5)
Evaluate(model_svm5, X_test_svm5, y_test_svm5, y_pred_svm5)
Evaluate(model_knn5, X_test_knn5, y_test_knn5, y_pred_knn5)

Evaluate(model_lr, X_test_lr, y_test_lr, y_pred_lr)
Evaluate(model_rf, X_test_rf, y_test_rf, y_pred_rf)
Evaluate(model_svm, X_test_svm, y_test_svm, y_pred_svm)
Evaluate(model_knn, X_test_knn, y_test_knn, y_pred_knn)

#LSTM
lstm_df = pd.read_csv('data/english_data.csv')
lstm_df.columns

hybrid_lstm_model, x_test_hlstm, y_test_hlstm, y_pred_hlstm = train_hybrid_lstm_model(lstm_df)
bd_lstm_model, x_test_bdlstm, y_test_bdlstm, y_pred_bdlstm = train_bd_lstm_model(lstm_df)

Evaluate(hybrid_lstm_model, x_test_hlstm, y_test_hlstm, y_pred_hlstm)
Evaluate(bd_lstm_model, x_test_bdlstm, y_test_bdlstm, y_pred_bdlstm)


'''Clustering'''
df_segment_80, df_segment_90 = Kmeans_pca(df_scaled)

print(classification_report(df_segment_80['label'], df_segment_80['Cluster'], target_names=['Human', 'Bot']))
cm = confusion_matrix(df_segment_80['label'], df_segment_80['Cluster'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Human', 'Bot'],
            yticklabels=['Human', 'Bot'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

print(classification_report(df_segment_90['label'], df_segment_90['Cluster'], target_names=['Human', 'Bot']))
cm = confusion_matrix(df_segment_90['label'], df_segment_90['Cluster'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Human', 'Bot'],
            yticklabels=['Human', 'Bot'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')