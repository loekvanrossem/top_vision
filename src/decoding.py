 # -*- coding: utf-8 -*-
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

EPSILON = 0.0000000000001


def shortest_cycle(graph, node2, node1):
    """
    Returns the shortest cycle going through an edge
    
    Used for computing weights in decode
    
    Parameters
    ----------
    graph: ndarray (n_nodes, n_nodes)
        A matrix containing the weights of the edges in the graph
    node1: int
        The index of the first node of the edge
    node2: int
        The index of the second node of the edge

    Returns
    -------
    cycle: list of ints
        A list of indices representing the nodes of the cycle in order
    """
    N = graph.shape[0]
    distances = np.inf * np.ones(N)
    distances[node2] = 0
    prev_nodes = np.zeros(N)
    prev_nodes[:] = np.nan
    prev_nodes[node2] = node1
    while (math.isnan(prev_nodes[node1])):
        distances_buffer = distances
        for j in range(N):
            possible_path_lengths = distances_buffer + graph[:,j]
            if (np.min(possible_path_lengths) < distances[j]):
                prev_nodes[j] = np.argmin(possible_path_lengths)
                distances[j] = np.min(possible_path_lengths)
    prev_nodes = prev_nodes.astype(int)
    cycle = [node1]
    while (cycle[0] != node2):
        cycle.insert(0,prev_nodes[cycle[0]])
    cycle.insert(0,node1)
    return cycle

def cohomological_parameterization(X ,cocycle_number=1, coeff=2,weighted=False):
    """
    Compute an angular parametrization on the data set corresponding to a given
    1-cycle
    
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
        as proposed in arxiv:1711.07205
    
    Returns
    -------
    decoding: ndarray(n_datapoints)
        The parameterization of the dataset consisting of a number between
        0 and 1 for each datapoint, to be interpreted modulo 1
    """
    # Get the cocycle
    result = ripser.ripser(X, maxdim=1, coeff=coeff, do_cocycles=True)
    diagrams = result['dgms']
    cocycles = result['cocycles']
    D = result['dperm2all']
    dgm1 = diagrams[1]
    idx = np.argsort(dgm1[:, 1] - dgm1[:, 0])[-cocycle_number] 
    cocycle = cocycles[1][idx]
    persistence(X, homdim=1, coeff=coeff, show_largest_homology=0,
                Nsubsamples=0, save_path=None, cycle=idx)
    thresh = dgm1[idx, 1]-EPSILON
    
    # Compute connectivity
    N = X.shape[0]
    connectivity = np.zeros([N,N])
    for i in range(N):
        for j in range(i):
            if D[i, j] <= thresh:
                connectivity[i,j] = 1
    cocycle_array = np.zeros([N,N])
    
    # Lift cocycle
    for i in range(cocycle.shape[0]):
        cocycle_array[cocycle[i,0],cocycle[i,1]] = (
            ((cocycle[i,2] + coeff/2) % coeff) - coeff/2
            )
        
    # Weights
    if (weighted):
        def real_cocycle(x):
            real_cocycle =(
                connectivity * (cocycle_array + np.subtract.outer(x, x))
                )
            return np.ravel(real_cocycle)
        
        # Compute graph
        x0 = np.zeros(N)
        res = least_squares(real_cocycle, x0)
        real_cocyle_array = res.fun
        real_cocyle_array = real_cocyle_array.reshape(N,N)
        real_cocyle_array = real_cocyle_array - np.transpose(real_cocyle_array)
        graph = np.array(real_cocyle_array>0).astype(float)
        graph[graph==0] = np.inf
        graph = (D + EPSILON) * graph   # Add epsilon to avoid NaNs
        
        # Compute weights
        cycle_counts = np.zeros([N,N])  
        iterator = trange(0, N, position=0, leave=True)
        iterator.set_description("Computing weights for decoding")
        for i in iterator:
            for j in range(N):
                if (graph[i,j] != np.inf):
                    cycle = shortest_cycle(graph, j, i)
                    for k in range(len(cycle)-1):
                        cycle_counts[cycle[k], cycle[k+1]] += 1
                
        weights = cycle_counts / (D + EPSILON)**2
        weights = np.sqrt(weights)
    else:
        weights = np.outer(np.ones(N),np.ones(N))
        
    def real_cocycle(x):
        real_cocycle =(
            weights * connectivity * (cocycle_array + np.subtract.outer(x, x))
            )
        return np.ravel(real_cocycle)
    
    # Smooth cocycle
    print("Decoding...", end=" ")
    x0 = np.zeros(N)
    res = least_squares(real_cocycle, x0)
    decoding = res.x
    decoding = np.mod(decoding, 1)
    print("done")
    
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
    decoding = decoding.to_numpy()[:,0]
    for i in range(X.shape[1]):
        effective_amplitude = cut_amplitude * (np.max(X[i]) - np.min(X[i]))
        cuts[:,i] = effective_amplitude * ((decoding - shift) % 1)
    reduced_data = X + cuts
    return reduced_data