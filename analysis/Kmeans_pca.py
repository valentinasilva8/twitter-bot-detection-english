# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 13:30:22 2025

"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='svg'
import seaborn as sns
#import kaleido


def Kmeans_pca(df_scaled):
    df =  df_scaled.drop('label', axis=1)
    
    pca = PCA()
    pca.fit(df)
    explained_variance = pca.explained_variance_ratio_
    
    # Plot explained variance
    plt.figure(figsize=(10,8))
    plt.plot(range(1, len(explained_variance)+1), explained_variance.cumsum(), marker='o')
    plt.xticks(range(1, len(explained_variance)+1))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by Components')
    plt.show()  
    
    
    '''Choosing components to preserve 80% of the variance'''
    # Final PCA with selected components
    pca_80 = PCA(n_components = 0.80)
    pca_scores_80 = pca_80.fit_transform(df)

    # Clustering
    kmeans_pca_80 = KMeans(n_clusters=2, init='k-means++', random_state=42)
    kmeans_pca_80.fit(pca_scores_80)
  
    
    # Create results dataframe
    df_segment_80 = pd.concat([df_scaled, 
                           pd.DataFrame(pca_scores_80, columns=[f'Component {i+1}' for i in range(pca_80.n_components_)])], 
                          axis=1)
    df_segment_80['Cluster'] = kmeans_pca_80.labels_
    #df_segment_80['Cluster'] = df_segment_80['Cluster'].replace({0:1, 1:0})
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_segment_80, x='Component 1', y='Component 2', hue='Cluster', palette='viridis',
                    s=50, alpha=0.6)
    plt.title('PCA Components with K-means Clustering')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.show()

    '''Choosing components to preserve 90% of the variance'''
    # Final PCA with selected components
    pca_90 = PCA(n_components = 0.90)
    pca_scores_90 = pca_90.fit_transform(df)

    # Clustering
    kmeans_pca_90 = KMeans(n_clusters=2, init='k-means++', random_state=42)
    kmeans_pca_90.fit(pca_scores_90)
  
    
    # Create results dataframe
    df_segment_90 = pd.concat([df_scaled, 
                           pd.DataFrame(pca_scores_90, columns=[f'Component {i+1}' for i in range(pca_90.n_components_)])], 
                          axis=1)
    df_segment_90['Cluster'] = kmeans_pca_90.labels_
    df_segment_90['Cluster'] = df_segment_90['Cluster'].replace({0:1, 1:0})
    
    '''3D Plot'''
    # Visualization
    PLOT = go.Figure()
    for C in sorted(df_segment_90.Cluster.unique()):
        PLOT.add_trace(go.Scatter3d(
            x = df_segment_90[df_segment_90.Cluster == C]['Component 1'],
            y = df_segment_90[df_segment_90.Cluster == C]['Component 2'],
            z = df_segment_90[df_segment_90.Cluster == C]['Component 3'],
            mode = 'markers',
            marker = dict(size=6, line=dict(width=1)),
            name = f'Cluster {C}'
        ))

    PLOT.update_traces(hovertemplate='Component 1: %{x}<br>Component 2: %{y}<br>Component 3: %{z}')

    PLOT.update_layout(
        width = 800, 
        height = 800, 
        title = '3D PCA Cluster Visualization',
        scene = dict(
            xaxis=dict(title='Component 1'),
            yaxis=dict(title='Component 2'),
            zaxis=dict(title='Component 3')
        ),
        font = dict(family="Gilroy", color='black', size=12)
    )

    PLOT.show()
    
    return df_segment_80, df_segment_90

    



    