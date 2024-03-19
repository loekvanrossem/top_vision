# -*- coding: utf-8 -*-
"""
Simulation of simple cells responding to grating images
"""
import numpy as np
from numpy.random import poisson
import pandas as pd
from scipy.integrate import dblquad
from itertools import product

from collections import namedtuple
from tqdm import trange
from numba import njit

import matplotlib.pyplot as plt

GRATING_PARS = ["orientation", "frequency", "phase", "contrast"]
Grating = namedtuple("Grating", GRATING_PARS)


@njit
def grating_function(x, y, grating):
    """Returns the value of a grating function at given x and y coordinates"""
    smallest_distance = 0.1
    theta, f, phi, C = grating
    return (
        C
        * np.exp(-1 / 2 * f**2 * smallest_distance**2)
        * np.cos(f * (x * np.cos(theta) + y * np.sin(theta)) + phi)
    )


def grating_image(grating, N=50, plot=True):
    """
    Make an image of a grating

    Parameters
    ----------
    grating: Grating
        A tuple containing the orientation, frequency, phase and contrast
    N: int, optional, default 50
        The number of pixels in each direction
    plot: bool, optional, default True
        When true plot the image

    Returns
    -------
    image: ndarray(N, N)
        An array of floats corresponding to the pixel values of the image
    """
    X = np.linspace(-1, 1, N)
    image = np.zeros([N, N])
    for i in range(0, N):
        for j in range(0, N):
            image[i, j] = grating_function(X[i], X[j], grating)
    if plot:
        plt.imshow(image, "gray", vmin=-1, vmax=1)
        plt.show()
    return image


def angular_mean(X, period=2 * np.pi, axis=None):
    """
    Average over an angular variable

    Parameters
    ----------
    X: ndarray
        Array of angles to average over
    period: float, optional, default 2*pi
        The period of the angles
    axis: int, optional, default None
        The axis of X to average over
    """
    ang_mean = (
        (period / (2 * np.pi))
        * np.angle(np.mean(np.exp((2 * np.pi / period) * X * 1j), axis=axis))
        % period
    )
    return ang_mean


@njit
def sigmoid(x):
    """Sigmoid function"""
    return 1 / (1 + np.exp(1 * (1 / 2 - x)))


def response(grating1, grating2, center=(0, 0)):
    """
    Neural response of a simple cell to a grating image

    Parameters
    ----------
    grating1: Grating
        Grating defining the simple cell
    grating2: Grating
        Grating corresponding to the image shown
    center: (float,float), optional, default (0,0)
        The focal point
    """
    fun1 = lambda s, t: grating_function(s + center[0], t + center[1], grating1)
    fun2 = lambda s, t: grating_function(s, t, grating2)
    product = lambda s, t: fun1(s, t) * fun2(s, t)
    integral = dblquad(product, -1, 1, -1, 1, epsabs=0.01)[0]
    response = sigmoid(integral)
    return response


def get_locations(N, random=False):
    """
    Return uniformly distributed locations on the grating parameter space

    Parameters
    ----------
    N: (int,int,int,int)
        The number of different orientations, frequencies, phases and contrasts
        respectively
    random: bool, optional, default False
        If true sample locations randomly as opposed to uniformly
    """
    N_or, N_fr, N_ph, N_co = N

    if random:
        sampling = np.random.uniform
    else:
        sampling = np.linspace

    if N_or == 1:
        orientation = [0]
    else:
        if N_ph == 1:
            orientation = sampling(0.0, 2 * (1 - 1 / N_or) * np.pi, N_or)
        else:
            orientation = sampling(0.0, (1 - 1 / N_or) * np.pi, N_or)

    if N_fr == 1:
        frequency = [3]
    else:
        frequency = sampling(0.0, 9, N_fr)

    if N_ph == 1:
        phase = [np.pi / 4]
    else:
        phase = sampling(0.0, 2 * (1 - 1 / N_ph) * np.pi, N_ph)

    if N_co == 1:
        contrast = [1]
    else:
        contrast = sampling(0.0, 1.0, N_co)

    locations = list(product(orientation, frequency, phase, contrast))
    return locations


def grating_model(
    Nn,
    Np,
    receptive_field_sigma=5,
    deltaT=None,
    random_focal_points=False,
    random_neurons=False,
    plot_stimuli=False,
):
    """
    Simulate the firing of simple cells responding to images of gratings

    Simple cells and stimuli are uniformly distributed along the parameter space

    Parameters
    ----------
    Nn: int
        The number of different orientations, frequencies, phases and contrasts
        for the neurons
        Directions where the stimuli do not vary will not be included
    Np: (int,int,int,int)
        The number of different orientations, frequencies, phases and contrasts
        respectively for the stimuli
    receptive_field_sigma: float, optional, default 5
        The width of the simple cell receptive fields
    deltaT: float, optional, default None
        The time period spikes are sampled over for each stimulus
        When None return the exact firing rates instead
    random_focal_points: bool, optional, default False
        If true randomize the focal point for each stimulus
    random_neurons: bool, optional, default False
        If true randomize the locations of the neurons
    plot_stimuli: bool, optional, default False
        If true plot an image of each stimulus
    average: int, optional, default=1
        The number of times the simulation is repeated and averaged over

    Returns
    -------
    data: dataframe(n_datapoints, n_neurons)
        The simulated firing rate data
    """

    Points = get_locations(Np)
    Neurons = get_locations(
        (
            [Nn, 1][Np[0] == 1],
            [Nn, 1][Np[1] == 1],
            [Nn, 1][Np[2] == 1],
            [Nn, 1][Np[3] == 1],
        ),
        random=random_neurons,
    )

    # Set focal points
    if random_focal_points:
        focal_points = np.random.random([len(Points), 2]) * 2 - 1
    else:
        focal_points = np.zeros([len(Points), 2])

    # Compute firing rates
    rates = np.zeros([len(Points), len(Neurons)])
    iterator = trange(0, len(Points), position=0, leave=True)
    iterator.set_description("Simulating data points")
    for i in iterator:
        if i % 1 == 0:
            if plot_stimuli:
                grating_image(Points[i])
        for j in range(0, len(Neurons)):
            rates[i, j] = response(Points[i], Neurons[j], center=focal_points[i])

    # Add noise
    if deltaT is None:
        data = rates
    else:
        data = poisson(rates * deltaT)

    data = pd.DataFrame(data)
    data = pd.merge(
        pd.DataFrame(Points, columns=GRATING_PARS),
        data,
        left_index=True,
        right_index=True,
    )
    data = data.set_index(GRATING_PARS)

    if deltaT is not None:
        print("Mean spike count: " + str(data.mean().mean()))

    return data
