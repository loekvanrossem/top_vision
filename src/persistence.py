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
import gudhi


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


def persistence_witness(X, number_of_landmarks=100, max_alpha_square=0.0, homdim=1):
    """
    Plot the persistence diagram of a dataset using gudhi

    Uses a witness complex allowing it to be used efficiently on larger datasets

    Parameters
    ----------
    X: dataframe(n_datapoints, n_features):
        Dataframe containing the data
    number_of_landmarks : int, optional, default 100
        The number of landmarks in the witness complex
    max_alpha_square : double, optional, default 0.0
        Maximal squared relaxation parameter
    homdim : int, optional, default 1
        The dimension of the homology
    """
    print("Sampling landmarks...", end=" ")

    witnesses = X.to_numpy()
    landmarks = gudhi.pick_n_random_points(
        points=witnesses, nb_points=number_of_landmarks
    )
    print("done")
    message = (
        "EuclideanStrongWitnessComplex with max_edge_length="
        + repr(max_alpha_square)
        + " - Number of landmarks="
        + repr(number_of_landmarks)
    )
    print(message)
    witness_complex = gudhi.EuclideanStrongWitnessComplex(
        witnesses=witnesses, landmarks=landmarks
    )
    simplex_tree = witness_complex.create_simplex_tree(
        max_alpha_square=max_alpha_square, limit_dimension=homdim
    )
    message = "Number of simplices=" + repr(simplex_tree.num_simplices())
    print(message)
    diag = simplex_tree.persistence()
    print("betti_numbers()=")
    print(simplex_tree.betti_numbers())
    gudhi.plot_persistence_diagram(diag, band=0.0)
    plt.show()
