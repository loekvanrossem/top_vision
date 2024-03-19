"""
Computational tools for persistence diagrams

Persistent homology from ripser and gudhi library
Confidence sets from arxiv:1303.7117
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
from tqdm import trange
import ripser
from persim import plot_diagrams


def hausdorff(data1: np.ndarray, data2: np.ndarray, homdim: int, coeff: int) -> float:
    """
    Compute the Hausdorff distance between two persistence diagrams.

    Parameters
    ----------
    data1 : np.ndarray
        The first persistence diagram.
    data2 : np.ndarray
        The second persistence diagram.
    homdim : int
        Homological dimension.
    coeff : int
        Coefficient field for homology.

    Returns
    -------
    float
        The Hausdorff distance between the two persistence diagrams.
    """
    dgm1 = ripser.ripser(data1, maxdim=homdim, coeff=coeff)["dgms"]
    dgm2 = ripser.ripser(data2, maxdim=homdim, coeff=coeff)["dgms"]
    distance = directed_hausdorff(dgm1[homdim], dgm2[homdim])[0]
    return distance


def confidence(
    X: pd.DataFrame,
    alpha: float = 0.05,
    Nsubsamples: int = 20,
    homdim: int = 1,
    coeff: int = 2,
) -> float:
    """
    Compute the confidence interval of the persistence diagram of a dataset.

    Computation done by subsampling as in arxiv:1303.7117

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe containing the data.
    alpha : float, optional
        1-alpha is the confidence level, default is 0.05.
    Nsubsamples : int, optional
        The number of subsamples, default is 20.
    homdim : int, optional
        The dimension of the homology, default is 1.
    coeff : int, optional
        The coefficient basis, default is 2.

    Returns
    -------
    float
        The confidence interval.
    """
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1.")
    if Nsubsamples < 1:
        raise ValueError("Nsubsamples must be at least 1.")
    N = X.shape[0]
    distances = np.zeros(Nsubsamples)
    iterator = trange(Nsubsamples, position=0, leave=True)
    iterator.set_description("Computing confidence interval")
    for i in iterator:
        subsample = X.sample(N, replace=True)
        distances[i] = hausdorff(X, subsample, homdim, coeff)
    distances.sort()
    confidence = np.sqrt(2) * 2 * distances[int(alpha * Nsubsamples)]
    return confidence


def persistence(
    X: pd.DataFrame,
    homdim: int = 1,
    coeff: int = 2,
    threshold: float = float("inf"),
    show_largest_homology: int = 0,
    distance_matrix: bool = False,
    Nsubsamples: int = 0,
    alpha: float = 0.05,
    cycle: int = None,
    save_path: str = None,
) -> None:
    """
    Plot the persistence diagram of a dataset using ripser.

    Also prints the five largest homology components.

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe containing the data.
    homdim : int, optional
        The dimension of the homology, default is 1.
    coeff : int, optional
        The coefficient basis, default is 2.
    threshold : float, optional
        The maximum distance in the filtration, default is infinity.
    show_largest_homology : int, optional
        Print this many of the largest homology components, default is 0.
    distance_matrix : bool, optional
        When true X will be interpreted as a distance matrix, default is False.
    Nsubsamples : int, optional
        The number of subsamples used in computing the confidence interval.
        Does not compute the confidence interval when this is 0, default is 0.
    alpha : float, optional
        1-alpha is the confidence level, default is 0.05.
    cycle : int, optional
        If given highlight the homology component in the plot corresponding to
        this cycle id, default is None.
    save_path : str, optional
        When given save the plot here, default is None.
    """
    result = ripser.ripser(
        X,
        maxdim=homdim,
        coeff=coeff,
        do_cocycles=True,
        distance_matrix=distance_matrix,
        thresh=threshold,
    )
    diagrams = result["dgms"]
    plot_diagrams(diagrams, show=False)

    if Nsubsamples > 0:
        conf = confidence(X, alpha, Nsubsamples, homdim, coeff)
        line_length = 10000
        plt.plot(
            [0, line_length],
            [conf, line_length + conf],
            color="green",
            linestyle="dashed",
            linewidth=2,
        )

    if cycle is not None:
        dgm1 = diagrams[1]
        plt.scatter(dgm1[cycle, 0], dgm1[cycle, 1], 20, "k", "x")

    if save_path is not None:
        path = f"{save_path}Z{coeff}"
        if Nsubsamples > 0:
            path += f"_confidence{1 - alpha}"
        path += ".png"
        plt.savefig(path)

    plt.show()

    if show_largest_homology != 0:
        dgm = diagrams[homdim]
        largest_indices = np.argsort(dgm[:, 0] - dgm[:, 1])
        largest_components = dgm[largest_indices[:show_largest_homology]]
        print(f"Largest {homdim}-homology components:")
        print(largest_components)
