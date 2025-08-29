# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 17:43:13 2025

@author: sqril
"""

import matplotlib.pyplot as plt
from preprocessing.SplittingDataset import SplittingDataset 
import seaborn as sns

def ExploratoryAnalysis(df):
    #value count of labels
    df['label'].map({0: 'Human', 1:'Bot'})
    label_count = df['label'].value_counts()
    label_count.plot(kind = 'bar', color = 'blue')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title('Count of Labels by Category')
    # add labels to the bars
    for i, count in enumerate(label_count):
        plt.text(i, count//2, str(count), ha='center', fontsize=9)
    plt.show()
    
    #Splitting Dataset for Analysis
    df_train, df_test = SplittingDataset(df)
        
    #Computing correlation with the label
    correlation = df_train.corr()['label'].abs().sort_values(ascending=False)
    correlation = correlation.drop('label')

    #Selecting top 5 features
    top_5_features = correlation.head(5).index.tolist()
        
    #heatmap
    plt.figure(figsize=(12,10))
    sns.heatmap(df_train.corr(), annot=True, cmap = "Greens")
    plt.title("Heatmap of Training Dataset")
    plt.show()
    
    #Reducing dataframe for pairplot
    df_reduced = df_train[top_5_features + ['label']].copy()
        
    #pairplot
    g = sns.pairplot(df_reduced, hue='label', plot_kws={'alpha':0.4 })
    g.fig.suptitle("Pairplot of Top 5 Training DataSet Attributes based on Correlation", y=1)
    plt.show()