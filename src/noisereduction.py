"""
A collection of noise reduction algorithms
"""

import numpy as np
import scipy
import pandas as pd

from sklearn.decomposition import PCA

from tqdm import trange
from numba import njit, prange

from plotting import plot_slider


@njit(parallel=True)
def _compute_gradient_F(
    S: np.ndarray, X: np.ndarray, sigma: float, omega: float
) -> np.ndarray:
    """
    Compute the gradient of F as described in arxiv:0910.5947.

    Parameters
    ----------
    S : np.ndarray
        Matrix S of shape (M, d) representing the subset constructing the denoised set.
    X : np.ndarray
        Matrix X of shape (N, d) representing the full dataset.
    sigma : float
        Parameter controlling the Gaussian kernel width.
    omega : float
        Strength of the repulsive force between datapoints.

    Returns
    -------
    gradF : np.ndarray
        Gradient of F with respect to the subset S, of shape (M, d).
    """
    gradF = np.zeros(S.shape)
    d = X.shape[1]
    N = X.shape[0]
    M = S.shape[0]
    for j in prange(M):
        normsSX = np.square(S[j] - X).sum(axis=1)
        normsSS = np.square(S[j] - S).sum(axis=1)
        expsSX = np.exp(-1 / (2 * sigma**2) * normsSX)
        expsSS = np.exp(-1 / (2 * sigma**2) * normsSS)
        SX, SS = np.zeros(d), np.zeros(d)
        for k in range(d):
            SX[k] = -1 / (N * sigma**2) * np.sum((S[j] - X)[:, k] * expsSX)
            SS[k] = omega / (M * sigma**2) * np.sum((S[j] - S)[:, k] * expsSS)
        gradF[j] = SX + SS
    return gradF


def top_noise_reduction(
    X: pd.DataFrame,
    n: int = 100,
    speed: float = 0.02,
    omega: float = 0.2,
    fraction: float = 0.1,
    plot_history: bool = False,
) -> pd.DataFrame:
    """
    Topological denoising algorithm as in arxiv:0910.5947.

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe containing the data.
    n : int, optional
        Number of iterations, default is 100.
    speed : float, optional
        The speed at which data points move during computation, default is 0.02.
    omega : float, optional
        Strength of the repulsive force between datapoints, default is 0.2.
    fraction : float, optional
        The fraction of datapoints from which the denoised dataset is constructed,
        default is 0.1.
    plot_history : bool, optional
        When true plot how the dataset looked during computation, default is False.

    Returns
    -------
    pd.DataFrame
        Denoised dataset.
    """
    if not 0 < fraction <= 1:
        raise ValueError("Fraction must be between 0 and 1.")

    n_plot_steps = 100
    N = X.shape[0]
    S = X.iloc[np.random.choice(N, round(fraction * N), replace=False)]
    sigma = X.stack().std()
    c = speed * np.max(scipy.spatial.distance.cdist(X, X, metric="euclidean"))
    history = [S]

    iterator = trange(0, n, position=0, leave=True)
    iterator.set_description("Topological noise reduction")
    for i in iterator:
        gradF = _compute_gradient_F(S.to_numpy(), X.to_numpy(), sigma, omega)

        if i == 0:
            maxgradF = np.max(np.sqrt(np.square(gradF).sum(axis=1)))
        S = S + c * gradF / maxgradF

        if plot_history:
            if i % np.ceil(n / n_plot_steps) == 0:
                history.append(S)

    if plot_history:
        plot_slider(history)

    return S


@njit(parallel=True)
def density_estimation(X: np.ndarray, k: int) -> np.ndarray:
    """
    Estimates density at each point.

    Parameters
    ----------
    X : np.ndarray
        Data points.
    k : int
        Density is estimated as 1 over the distance to the k-th nearest point.

    Returns
    -------
    np.ndarray
        Array containing density estimates for each point.
    """
    N = X.shape[0]
    densities = np.zeros(N)
    for i in range(N):
        distances = np.sum((X[i] - X) ** 2, axis=1)
        densities[i] = 1 / np.sort(distances)[k]
    return densities


def density_filtration(X: pd.DataFrame, k: int, fraction: float) -> pd.DataFrame:
    """
    Returns the points with highest density.

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe containing the data.
    k : int
        Density is estimated as 1 over the distance to the k-th nearest point.
    fraction : float
        The fraction of highest density datapoints that are returned.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the points with highest density.
    """
    if not (0 <= fraction <= 1):
        raise ValueError("Fraction must be between 0 and 1.")

    if not (0 < k <= X.shape[0]):
        raise ValueError(
            "k must be greater than 0 and less than or equal to the number of data points."
        )

    print("Applying density filtration...")
    X["densities"] = density_estimation(X.to_numpy().astype(np.float), k)
    X = X.nlargest(int(fraction * X.shape[0]), "densities")
    X = X.drop(columns="densities")
    print("Density filtration complete.")
    return X


@njit(parallel=True)
def _compute_averages(X: np.ndarray, r: float) -> np.ndarray:
    """
    Compute neighborhood averages.

    Parameters
    ----------
    X : np.ndarray
        Data points.
    r : float
        Radius within which points are averaged.

    Returns
    -------
    np.ndarray
        Array containing neighborhood averages for each point.
    """
    N = X.shape[0]
    averages = np.zeros(X.shape)
    for i in prange(N):
        distances = np.sum((X[i] - X) ** 2, axis=1)
        neighbors = X[distances < r]
        averages[i] = np.sum(neighbors, axis=0) / len(neighbors)
    return averages


def neighborhood_average(X: pd.DataFrame, r: float) -> pd.DataFrame:
    """
    Replace each point by an average over its neighborhood.

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe containing the data.
    r : float
        Points are averaged over all points within radius r.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the points averaged over their neighborhoods.
    """
    if r <= 0:
        raise ValueError("Radius 'r' must be greater than 0.")

    print("Applying neighborhood average...")
    averages = _compute_averages(X.to_numpy().astype(np.float), r)
    print("Neighborhood average applied.")

    return pd.DataFrame(data=averages, index=X.index, columns=X.columns)


def z_cutoff(X: pd.DataFrame, z_cutoff: float) -> pd.DataFrame:
    """
    Remove outliers with a high Z-score.

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe containing the data.
    z_cutoff : float
        The Z-score at which points are removed.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the data after removing outliers.
    """
    if z_cutoff < 0:
        raise ValueError("z_cutoff must be non-negative.")

    z_scores = np.abs(scipy.stats.zscore(X))
    result = X[(z_scores < z_cutoff).all(axis=1)]
    removed_count = len(X) - len(result)

    print(f"{removed_count} datapoints with Z-score above {z_cutoff} removed.")
    return result


def PCA_reduction(X: pd.DataFrame, dim: int) -> (pd.DataFrame, np.ndarray):
    """
    Use principal component analysis to reduce the data to a lower dimension.

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe containing the data.
    dim : int
        The number of dimensions the data is reduced to.

    Returns
    -------
    X_reduced : pd.DataFrame
        Dataframe containing the data after PCA reduction.
    """
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError("dim must be a positive integer.")

    if dim > X.shape[1]:
        raise ValueError("dim cannot exceed the number of features in the dataset.")

    pca = PCA(n_components=dim)
    pca.fit(X)
    X = pd.DataFrame(pca.transform(X), columns=[i for i in range(dim)], index=X.index)
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    return X
