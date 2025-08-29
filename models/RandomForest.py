# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 17:48:10 2025

"""
'''Random Forest Model'''

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN
from joblib import parallel_backend


def RandomForest(df_train, df_test):
    X_train = df_train.drop(['label'], axis=1)
    y_train = df_train['label']
    X_test = df_test.drop(['label'],axis=1)
    y_test = df_test['label']
    
    # Apply SMOTe for class balancing
    smote = SMOTEENN(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Define parameter grid for RandomizedSearchCV
    param_grid = {
        'n_estimators': [200, 500, 1000],
        'max_depth': [10, 20, 30,],
        'min_samples_split': [5, 10,20],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    # Initialize Random Forest Classifier
    rf = RandomForestClassifier(random_state=42, class_weight={0: 3, 1: 1})

    # Use RandomizedSearchCV for tuning
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=20,  # Try 10 parameter combinations
        cv=5,       # 5-fold CV
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    # Fit the model with parallel backend
    with parallel_backend('threading', n_jobs=-1):
        random_search.fit(X_train_resampled, y_train_resampled)

    # Get best model
    best_model = random_search.best_estimator_

    # Make predictions
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Apply threshold adjustment for prediction
    threshold = 0.6
    y_pred_rf = (y_proba >= threshold).astype(int)

    return best_model, X_test, y_test, y_pred_rf