# -*- coding: utf-8 -*-
"""
A collection of noise reduction algorithms
"""
import numpy as np
import scipy
import pandas as pd

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA

from tqdm import trange
from numba import njit, prange

from persistence import persistence
from decorators import multi_input
from plotting import plot_slider


@njit(parallel=True)
def compute_gradient(S, X, sigma, omega):
    """Compute gradient of F as in arxiv:0910.5947"""
    gradF = np.zeros(S.shape)
    d = X.shape[1]
    N = X.shape[0]
    M = S.shape[0]
    for j in range(0,M):
        normsSX = np.square(S[j] - X).sum(axis=1)
        normsSS = np.square(S[j] - S).sum(axis=1)
        expsSX = np.exp(-1/(2*sigma**2)*normsSX)
        expsSS = np.exp(-1/(2*sigma**2)*normsSS)
        SX, SS = np.zeros(d), np.zeros(d)
        for k in range(0,d):
            SX[k] = -1/(N*sigma**2) * np.sum((S[j] - X)[:,k] * expsSX)
            SS[k] = omega/(M*sigma**2) * np.sum((S[j] - S)[:,k] * expsSS)
        gradF[j] = SX + SS
    return gradF

@multi_input
def top_noise_reduction(X, n=100, speed=0.02, omega=0.2, fraction=0.1, plot_history=False):
    """
    Topological denoising algorithm as in arxiv:0910.5947
    
    Parameters
    ----------
    X: dataframe(n_datapoints, n_features):
        Dataframe containing the data
    n: int, optional, default 100
        Number of iterations
    speed: float, optional, default 0.02
        The speed at which data points move during computation
    omega: float, optional, default 0.2
        Strength of the repulsive force between datapoints
    fraction: float between 0 and 1, optional, default 0.1
        The fraction of datapoints from which the denoised dataset is
        constructed
    plot_history: bool, optional, default False
        When true plot how the datset looked during computation
    """
    n_plot_steps = 100
    N = X.shape[0]
    S = X.iloc[np.random.choice(N, round(fraction*N), replace=False)]
    sigma = X.stack().std()
    c = speed * np.max(scipy.spatial.distance.cdist(X, X, metric='euclidean'))
    history = [S]

    iterator = trange(0, n, position=0, leave=True)
    iterator.set_description("Topological noise reduction")
    for i in iterator:
        gradF = compute_gradient(S.to_numpy(), X.to_numpy(), sigma, omega)
            
        if i == 0:
            maxgradF = np.max(np.sqrt(np.square(gradF).sum(axis=1)))
        S = S + c* gradF/maxgradF

        if plot_history:
            if i % np.ceil(n/n_plot_steps) == 0:
                history.append(S)
    
    if plot_history:
        plot_slider(history)

    return S

@njit(parallel=True)
def density_estimation(X,k):
    """Estimates density at each point"""
    N = X.shape[0]
    densities = np.zeros(N)
    for i in prange(N):
        distances = np.sum((X[i] - X)**2, axis=1)
        densities[i] = 1/np.sort(distances)[k]
    return densities

@multi_input
def density_filtration(X, k, fraction):
    """
    Returns the points which are in locations with high density

    Parameters
    ----------
    X: dataframe(n_datapoints, n_features):
        Dataframe containing the data
    k: int
        Density is estimated as 1 over the distance to the k-th nearest point
    fraction: float between 0 and 1
        The fraction of highest density datapoints that are returned
    """
    print("Applying density filtration...", end=" ")
    N = X.shape[0]
    X["densities"] = density_estimation(X.to_numpy().astype(np.float),k)
    X = X.nlargest(int(fraction * N), "densities")
    X = X.drop(columns="densities")
    print("done")
    return X

@njit(parallel=True)
def compute_averages(X, r):
    """Used in neighborhood_average"""
    N = X.shape[0]
    averages = np.zeros(X.shape)
    for i in prange(N):
        distances = np.sum((X[i] - X)**2, axis=1)
        neighbors = X[distances < r]
        averages[i] = np.sum(neighbors, axis=0)/len(neighbors)
    return averages

@multi_input
def neighborhood_average(X, r):
    """
    Replace each point by an average over its neighborhood

    Parameters
    ----------
    X: dataframe(n_datapoints, n_features):
        Dataframe containing the data
    r : float
        Points are averaged over all points within radius r
    """
    print("Applying neighborhood average...", end=" ")
    averages = compute_averages(X.to_numpy().astype(np.float),r)
    print("done")
    result = pd.DataFrame(data=averages,index=X.index)
    return result

@multi_input
def z_cutoff(X, z_cutoff):
    """
    Remove outliers with a high Z-score

    Parameters
    ----------
    X: dataframe(n_datapoints, n_features):
        Dataframe containing the data
    z_cutoff : float
        The Z-score at which points are removed
    """
    z=np.abs(scipy.stats.zscore(np.sqrt(np.square(X).sum(axis=1))))
    result = X[(z < z_cutoff)]
    print(f"{len(X) - len(result)} datapoints with Z-score above {z_cutoff}"
          + " removed")
    return result

@multi_input
def PCA_reduction(X, dim):
    """
    Use principle component analysis to reduce the data to a lower dimension
    
    Also print the variance explained by each component
    Parameters
    ----------
    X: dataframe(n_datapoints, n_features):
        Dataframe containing the data
    dim : int
        The number of dimensions the data is reduced to
    """
    pca = PCA(n_components=dim)
    pca.fit(X)
    columns = [i for i in range(dim)]
    X = pd.DataFrame(pca.transform(X), columns=columns, index=X.index)
    print("PCA explained variance:")
    print(pca.explained_variance_ratio_)
    return X