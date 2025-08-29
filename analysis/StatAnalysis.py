# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 17:33:08 2025

"""
'''Statistics about the dataset'''

def StatAnalysis(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    print("Mean:")
    print(numeric_df.mean().round(2))
    print("\n")
    print("Median:")
    print(numeric_df.median().round(2))
    print("\n")
    print("Mode:")
    print(numeric_df.mode().round(2))
    print("\n")
    print("Standard Deviation:")
    print(numeric_df.std().round(2))
    print("\n")
    print("Variance:")
    print(numeric_df.var().round(2))
    print("\n")
    print("Minimum:")
    print(numeric_df.min().round(2))
    print("\n")
    print("Maximum:")
    print(numeric_df.max().round(2))
