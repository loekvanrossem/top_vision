# -*- coding: utf-8 -*-
import numpy as np

from tqdm import trange

import matplotlib.pyplot as plt

from decorators import multi_input


@multi_input
def estimate_dimension(X, max_size, test_size = 30, Nsteps = 20, fraction = 0.5):
    """
    Plots an estimation of the dimension of a dataset at different scales

    Parameters
    ----------
    X: dataframe(n_datapoints, n_features):
        Dataframe containing the data
    max_size : float
        The upper bound for the scale
    test_size : int, optional, default 30
        The number of datapoints used to estimate the density
    Nsteps : int, optional, default 20
        The number of different scales at which the density is estimated
    fraction : float between 0 and 1, optional, default 0.5
        Difference in radius between the large sphere and smaller sphere used to compute density

    Returns
    -------
    average : ndarray(Nsteps)
        The dimension at each scale

    """
    average = np.zeros(Nsteps)
    S = X.iloc[np.random.choice(X.shape[0], test_size, replace=False)]
    
    iterator = trange(0, Nsteps, position=0, leave=True)
    iterator.set_description("Estimating dimension")
    for n in iterator:
        size = max_size*n/Nsteps
        count_small = np.zeros(X.shape[0])
        count_large = np.zeros(X.shape[0])
        dimension = np.zeros(S.shape[0])
        for i in range(0,S.shape[0]):
            for j in range(0,X.shape[0]):
                distance = np.sqrt(np.square(S.iloc[i] - X.iloc[j]).sum())
                if (distance < size/fraction):
                    count_large[i] += 1
                if (distance < size):
                    count_small[i] += 1
            if (count_large[i] != 0):
                dimension[i] = np.log(count_small[i]/count_large[i])/np.log(fraction)
            else:
                dimension[i] = 0
        average[n] = np.mean(dimension)
    plt.plot(range(0, Nsteps), average)
    plt.xlabel("Scale")
    plt.ylabel("Dimension")
    plt.show()
    return average