import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Standard Plotting

def plot_clusters(X, y, dim, points,
                labels_prefix = 'cluster', 
                points_name = 'centroids',
                colors = cm.gist_rainbow,
                    # https://matplotlib.org/examples/color/colormaps_reference.html
#                   colors = ['brown', 'orange', 'olive', 
#                             'green', 'cyan', 'blue', 
#                             'purple', 'pink'],
#                   points_color = 'red'
                points_color = 'cyan'
                ):
    """
    Plot a two dimensional projection of an array of labelled points
    X:      array with at least two columns
    y:      vector of labels, length as number of rows in X
    dim:    the two columns to project, inside range of X columns, e.g. (0,1)
    points: additional points to plot as 'stars'
    labels_prefix: prefix to the labels for the legend ['cluster']
    points_name:   legend name for the additional points ['centroids']
    colors: a color map
    points_color: the color for the points
    """
    # plot the labelled (colored) dataset and the points
    labels = np.unique(y)
    for i in range(len(labels)):
        color = colors(i / len(labels)) # choose a color from the map
        plt.scatter(X[y==labels[i],dim[0]], 
                    X[y==labels[i],dim[1]], 
                    s=10, 
                    c = [color], # scatter requires a sequence of colors
                    marker='s', 
                    label=labels_prefix+str(labels[i]))
    if points is not None:
        plt.scatter(points[:,dim[0]], 
                    points[:,dim[1]], 
                    s=50, 
                    marker='*', 
                    c=[points_color], 
                    label=points_name)
    # plt.legend()
    plt.grid()
    plt.show()

def cluster_gonzalez2 (X, num_clusters):
    # Initialization
    heads = np.zeros(num_clusters, dtype=int)
    B = []
    for i in range(num_clusters):
        B.append([])
    B[0]=list(range(X.shape[0]))
    
    # Expansion phase
    for l in range(num_clusters-1):
        h = 0
        index = -1
        to_pop = -1
        for j in range(l+1):
    # Finding most distant element from current head
            for vi in B[j]:
                distance = point_euclidean_distance(X[heads[j]], X[vi])
                if distance > h:
                    h = distance
                    index = vi
                    to_pop = j
    # Reassigning the most distant element to the new cluster and setting it as the head
        B[to_pop].remove(index)
        B[l+1].append(index)
        heads[l+1] = index
    
        for j in range(l+1):
            to_remove = []
            for vt in B[j]:
                if point_euclidean_distance(X[vt], X[heads[j]])>= point_euclidean_distance(X[vt], X[index]):
                    to_remove.append(vt)
            for vt in to_remove:
                B[j].remove(vt)
                B[l+1].append(vt)
    return B


def cluster_gonzalezMedoid (X, num_clusters):
    # Initialization
    heads = np.zeros(num_clusters, dtype=int)
    B = []
    for i in range(num_clusters):
        B.append([])
    B[0]=list(range(X.shape[0]))
    
    # Expansion phase
    for l in range(num_clusters-1):
        h = 0
        index = -1
        to_pop = -1
        for j in range(l+1):
    # Finding most distant element from current head
            for vi in B[j]:
                distance = point_euclidean_distance(X[heads[j]], X[vi])
                if distance > h:
                    h = distance
                    index = vi
                    to_pop = j
    # Reassigning the most distant element to the new cluster and setting it as the head
        B[to_pop].remove(index)
        B[l+1].append(index)
        heads[l+1] = index
    
        for j in range(l+1):
            to_remove = []
            for vt in B[j]:
                if point_euclidean_distance(X[vt], X[heads[j]])>= point_euclidean_distance(X[vt], X[index]):
                    to_remove.append(vt)
            for vt in to_remove:
                B[j].remove(vt)
                B[l+1].append(vt)
    
    return heads


def point_euclidean_distance(p1, p2):
    if isinstance(p1,np.int32):
        return(abs(p1-p2))
    return math.sqrt(sum(np.square(p1-p2)))