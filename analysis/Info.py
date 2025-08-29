# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 17:32:28 2025

@author: sqril
"""

''' Information about Dataframe'''

def Info(df):
    print("Info:")
    print(df.info())
    print("\n")
    print("Head:")
    print(df.head())
    print("\n")
    print("Tail:")
    print(df.tail())
    print("\n")
    print("Shape:")
    print(df.shape)
    print("\n")
    print("Size:")
    print(df.size)
    print("\n")
    print("Null Count:")
    print(df.isnull().sum())
    print("\n")