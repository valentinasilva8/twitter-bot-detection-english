# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 17:52:52 2025

@author: sqril
"""
'''Evaluating Models'''

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def EvaluateModel(model, X_test, y_test, y_pred):
    # Get the model's class name
    model_name = model.__class__.__name__
    
    #Evaluating Model
    roc_auc = roc_auc_score(y_test, y_pred)
    
    print(f"{model_name} roc-auc: {roc_auc:.2f}")
    
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Human', 'Bot']))
    
    if model_name in ('RandomForestClassifier'):
        # Create visualizations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Human', 'Bot'],
                    yticklabels=['Human', 'Bot'],
                    ax=ax1)
        ax1.set_title(f'{model_name} Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Feature Importance (top 5 only)
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head()
        
        sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax2)
        ax2.set_title('Top 5 Important Features')
        
        plt.tight_layout()
        plt.show()
        
    else:
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Human', 'Bot'],
                    yticklabels=['Human', 'Bot'])
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
            