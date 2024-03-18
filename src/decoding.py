"""
Use cohomology to decode datasets with circular parameters

Persistent homology from arxiv:1908.02518
Homological decoding from DOI:10.1007/s00454-011-9344-x and arxiv:1711.07205
"""

import math
import numpy as np
from scipy.optimize import least_squares
import pandas as pd

from tqdm import trange

import ripser

from persistence import persistence

EPSILON = 1e-15


def shortest_cycle(weight_matrix, start_node, end_node):
    """
    Finds the shortest cycle passing through an edge in a graph represented by its weight matrix.

    Parameters
    ----------
    weight_matrix: ndarray (n_nodes, n_nodes)
        A matrix containing the weights of the edges in the graph.
    start_node: int
        The index of the first node of the edge.
    end_node: int
        The index of the second node of the edge.

    Returns
    -------
    cycle: list of ints
        A list of indices representing the nodes of the cycle in order.
    """
    N = weight_matrix.shape[0]
    distances = np.full(N, np.inf)
    distances[end_node] = 0
    prev_nodes = np.full(N, np.nan)
    prev_nodes[end_node] = start_node

    # Floyd-Warshall algorithm for finding all-pairs shortest paths
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if distances[i] + weight_matrix[i, j] < distances[j]:
                    distances[j] = distances[i] + weight_matrix[i, j]
                    prev_nodes[j] = i

    cycle = [end_node]
    current_node = end_node
    while current_node != start_node:
        current_node = int(prev_nodes[current_node])
        cycle.insert(0, current_node)
    cycle.append(start_node)

    return cycle


def cohomological_parameterization(X, cocycle_number=1, coeff=2, weighted=False):
    """
    Compute an angular parametrization on the data set corresponding to a given 1-cycle.

    Parameters
    ----------
    X: ndarray(n_datapoints, n_features):
        Array containing the data
    cocycle_number: int, optional, default 1
        An integer specifying the 1-cycle used
        The n-th most stable 1-cycle is used, where n = cocycle_number
    coeff: int prime, optional, default 1
        The coefficient basis in which we compute the cohomology
    weighted: bool, optional, default False
        When true use a weighted graph for smoother parameterization

    Returns
    -------
    decoding: DataFrame
        The parameterization of the dataset consisting of a number between
        0 and 1 for each datapoint, to be interpreted modulo 1
    """
    # Compute persistent homology
    result = ripser.ripser(X, maxdim=1, coeff=coeff, do_cocycles=True)
    diagrams, cocycles, D = result["dgms"], result["cocycles"], result["dperm2all"]
    dgm1 = diagrams[1]

    # Select the cocycle to use
    idx = np.argsort(dgm1[:, 1] - dgm1[:, 0])[-cocycle_number]
    cocycle = cocycles[1][idx]

    # Determine threshold
    persistence(
        X,
        homdim=1,
        coeff=coeff,
        show_largest_homology=0,
        Nsubsamples=0,
        save_path=None,
        cycle=idx,
    )
    thresh = dgm1[idx, 1] - EPSILON

    # Construct connectivity matrix based on threshold
    connectivity = np.tril(D <= thresh, k=-1).astype(int)

    # Initialize array to store cocycle values
    N = X.shape[0]
    cocycle_array = np.zeros((N, N))
    for i, (a, b, c) in enumerate(cocycle):
        cocycle_array[a, b] = (c + coeff / 2) % coeff - coeff / 2

    if weighted:
        # Initialize weights matrix
        weights = np.zeros_like(connectivity)

        # Compute weights for each edge
        for i in trange(N, desc="Computing weights for decoding"):
            for j in range(N):
                if connectivity[i, j] != 0:
                    cycle = shortest_cycle(connectivity, j, i)
                    for k in range(len(cycle) - 1):
                        weights[cycle[k], cycle[k + 1]] += 1

        # Normalize weights and take square root
        weights /= (D + EPSILON) ** 2
        weights = np.sqrt(weights)
    else:
        weights = np.ones_like(connectivity)

    # Apply smoothing
    def real_cocycle(x):
        return np.ravel(
            weights * connectivity * (cocycle_array + np.subtract.outer(x, x))
        )

    x0 = np.zeros(N)
    res = least_squares(real_cocycle, x0)

    # Convert decoding results to DataFrame
    decoding = np.mod(res.x, 1)
    decoding = pd.DataFrame(decoding, columns=["decoding"])
    decoding = decoding.set_index(X.index)

    return decoding


def remove_feature(X, decoding, shift=0, cut_amplitude=1.0):
    """
    Removes a decoded feature from a dataset by making a cut at a fixed value
    of the decoding

    Parameters
    ----------
    X: dataframe(n_datapoints, n_features):
        Array containing the data
    decoding : dataframe(n_datapoints)
        The decoded feature, assumed to be angular with periodicity 1
    shift : float between 0 and 1, optional, default 0
        The location of the cut
    cut_amplitude : float, optional, default 1
        Amplitude of the cut
    """
    cuts = np.zeros(X.shape)
    decoding = decoding.to_numpy()[:, 0]
    for i in range(X.shape[1]):
        effective_amplitude = cut_amplitude * (np.max(X[i]) - np.min(X[i]))
        cuts[:, i] = effective_amplitude * ((decoding - shift) % 1)
    reduced_data = X + cuts
    return reduced_data
