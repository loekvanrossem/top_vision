# -*- coding: utf-8 -*-
"""
Tools to compute persistence diagrams

Persistent homology from ripser and gudhi library
Confidence sets from arxiv:1303.7117
"""
import numpy as np
from scipy.spatial.distance import directed_hausdorff

import matplotlib.pyplot as plt

from tqdm import trange

import ripser
from persim import plot_diagrams
import gudhi

from decorators import multi_input

    
def hausdorff(data1, data2, homdim, coeff):
    """Hausdorff metric between two persistence diagrams"""
    dgm1 = (ripser.ripser(data1,maxdim=homdim,coeff=coeff))['dgms']
    dgm2 = (ripser.ripser(data2,maxdim=homdim,coeff=coeff))['dgms']
    distance = directed_hausdorff(dgm1[homdim], dgm2[homdim])[0]
    return distance

@multi_input
def confidence(X, alpha=0.05, Nsubsamples=20, homdim=1, coeff=2):
    """
    Compute the confidence interval of the persistence diagram of a dataset
    
    Computation done by subsampling as in arxiv:1303.7117
    
    Parameters
    ----------
    X: dataframe(n_datapoints, n_features):
        Dataframe containing the data
    alpha : float between 0 and 1, optional, default 0.05
        1-alpha is the confidence
    Nsubsamples : int, optional, default 20
        The number of subsamples
    homdim : int, optional, default 1
        The dimension of the homology
    coeff : int prime, optional, default 2
        The coefficient basis
    """
    N = X.shape[0]
    distances = np.zeros(Nsubsamples)
    iterator = trange(0, Nsubsamples, position=0, leave=True)
    iterator.set_description("Computing confidence interval")
    for i in iterator:
        subsample = X.iloc[np.random.choice(N, N, replace=True)]
        distances[i] = hausdorff(X, subsample, homdim, coeff)
    distances.sort()
    confidence = np.sqrt(2) * 2 * distances[int(alpha*Nsubsamples)]
    return confidence

@multi_input
def persistence(X, homdim=1, coeff=2, threshold=float('inf'),
                show_largest_homology=0, distance_matrix=False, Nsubsamples=0,
                alpha=0.05, cycle=None, save_path=None):
    """
    Plot the persistence diagram of a dataset using ripser

    Also prints the five largest homology components
    
    Parameters
    ----------
    X: dataframe(n_datapoints, n_features):
        Dataframe containing the data
    homdim : int, optional, default 1
        The dimension of the homology
    coeff : int prime, optional, default 2
        The coefficient basis
    threshold : float, optional, default infinity
        The maximum distance in the filtration
    show_largest_homology: int, optional, default 0
        Print this many of the largest homology components
    distance_matrix : bool, optional, default False
        When true X will be interepreted as a distance matrix
    Nsubsamples : int, optional, default 0
        The number of subsamples used in computing the confidence interval
        Does not compute the confidence interval when this is 0
    alpha : float between 0 and 1, optional, default 0.05
        1-alpha is the confidence
    cycle : int, optional, default None
        If given highlight the homology component in the plot corresponding to
        this cycle id
    save_path : str, optional, default None
        When given save the plot here
    """
    result = ripser.ripser(X, maxdim=homdim, coeff=coeff, do_cocycles=True,
                           distance_matrix=distance_matrix, thresh=threshold)
    diagrams = result['dgms']
    plot_diagrams(diagrams, show=False)
    if (Nsubsamples>0):
        conf = confidence(X, alpha, Nsubsamples, homdim, 2)
        line_length = 10000
        plt.plot([0, line_length], [conf, line_length + conf], color='green',
                 linestyle='dashed',linewidth=2)
    if cycle is not None:
        dgm1 = diagrams[1]
        plt.scatter(dgm1[cycle, 0], dgm1[cycle, 1], 20, 'k', 'x')
    if save_path is not None:
        path = save_path + 'Z' + str(coeff)
        if (Nsubsamples>0):
            path += '_confidence' + str(1-alpha)
        path += '.png'
        plt.savefig(path)
    plt.show()
    
    if show_largest_homology != 0:
        dgm = diagrams[homdim]
        largest_indices = np.argsort(dgm[:, 0] - dgm[:, 1])
        largest_components = dgm[largest_indices[:show_largest_homology]]
        print(f"Largest {homdim}-homology components:")
        print(largest_components)
    return

@multi_input
def persistence_witness(X, number_of_landmarks=100, max_alpha_square=0.0,
                        homdim=1):
    """
    Plot the persistence diagram of a dataset using gudhi

    Uses a witness complex allowing it to be used on larger datasets
    
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
        max_alpha_square=max_alpha_square,
        limit_dimension=homdim
    )
    message = "Number of simplices=" + repr(simplex_tree.num_simplices())
    print(message)
    diag = simplex_tree.persistence()
    print("betti_numbers()=")
    print(simplex_tree.betti_numbers())
    gudhi.plot_persistence_diagram(diag, band=0.0)
    plt.show()
    return