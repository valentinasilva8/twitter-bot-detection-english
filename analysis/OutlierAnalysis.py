# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 17:35:35 2025

"""

'''Outlier Analysis'''

import numpy as np

def OutlierAnalysis(df):
    df = df.dropna() #drop null values
    df1 = df.select_dtypes(include=['float64', 'int64']) #dropping nonnumeric columns columns
    
    q1 = df1.quantile(.25)
    q3 = df1.quantile(.75)
    iqr = q3 -q1
    
    #calculating upper and lower bound
    LowerBound=q1-1.5*iqr
    UpperBound=q3+1.5*iqr

    print("The value of lower bound is:",round(LowerBound),sep='\n')
    print("\n")
    print("The value of upper bound is:",round(UpperBound),sep='\n')
    print("\n")
    
    #getting number of outliers
    for i in df1.columns:
        outlier = np.array(np.where(((df1[i]) < LowerBound[i]) | ((df1[i]) > UpperBound[i])))
        print("The number of outliers for ", i, " is ", outlier.size)
    print("\n")