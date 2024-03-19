"""
Simulation of simple cells responding to grating images
"""

from typing import Optional
from collections import namedtuple
from tqdm import trange

from numba import njit
import numpy as np
from numpy.random import poisson
import pandas as pd
from scipy.integrate import dblquad
from itertools import product


import matplotlib.pyplot as plt

GRATING_PARAMS = ["orientation", "frequency", "phase", "contrast"]
Grating = namedtuple("Grating", GRATING_PARAMS)


@njit
def grating_function(x, y, grating):
    """Returns the value of a grating function at given x and y coordinates"""
    theta, f, phi, C = grating
    return C * np.cos(f * (x * np.cos(theta) + y * np.sin(theta)) + phi)


def grating_image(grating, N=50, plot=True):
    """
    Make an image of a grating.

    Parameters
    ----------
    grating : Grating
        A tuple containing the orientation, frequency, phase, and contrast.
    N : int, optional
        The number of pixels in each direction, default is 50.
    plot : bool, optional
        When true, plot the image, default is True.

    Returns
    -------
    image : ndarray(N, N)
        An array of floats corresponding to the pixel values of the image.
    """
    X = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(X, X)
    image = grating_function(X, Y, grating)

    if plot:
        plt.figure(figsize=(1, 1))
        plt.imshow(image, cmap="gray", vmin=-1, vmax=1)
        plt.show()

    return image


def angular_mean(
    X: np.ndarray, period: float = 2 * np.pi, axis: int = None
) -> np.ndarray:
    """
    Average over an angular variable.

    Parameters
    ----------
    X : np.ndarray
        Array of angles to average over.
    period : float, optional
        The period of the angles, default is 2*pi.
    axis : int, optional
        The axis of X to average over, default is None.

    Returns
    -------
    np.ndarray
        Angular mean.
    """
    ang_mean = (
        (period / (2 * np.pi))
        * np.angle(np.mean(np.exp((2 * np.pi / period) * X * 1j), axis=axis))
        % period
    )
    return ang_mean


@njit
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid function."""
    return 1 / (1 + np.exp(1 * (1 / 2 - x)))


def response(
    grating1: Grating, grating2: Grating, center: tuple[float, float] = (0, 0)
) -> float:
    """
    Neural response of a simple cell to a grating image.

    Parameters
    ----------
    grating1 : Grating
        Grating defining the simple cell.
    grating2 : Grating
        Grating corresponding to the image shown.
    center : Tuple[float, float], optional
        The focal point, default is (0,0).

    Returns
    -------
    float
        Neural response.
    """

    def fun1(s: float, t: float) -> float:
        """Grating function for grating1."""
        return grating_function(s + center[0], t + center[1], grating1)

    def fun2(s: float, t: float) -> float:
        """Grating function for grating2."""
        return grating_function(s, t, grating2)

    def product(s: float, t: float) -> float:
        """Product of grating functions."""
        return fun1(s, t) * fun2(s, t)

    integral = dblquad(product, -1, 1, -1, 1, epsabs=0.01)[0]
    return sigmoid(integral)


def get_locations(N, random=False):
    """
    Generate uniformly or randomly distributed locations in the grating parameter space.

    Parameters
    ----------
    N : (int,int,int,int)
        The number of different orientations, frequencies, phases, and contrasts respectively.
    random : bool, optional
        If True, sample locations randomly; otherwise, sample uniformly. Default is False.

    Returns
    -------
    list of tuple
        List of tuples representing locations in the parameter space.

    Raises
    ------
    ValueError
        If any of the input values in N are non-positive integers.
    """
    # Validate input parameters
    if not all(isinstance(n, int) and n > 0 for n in N):
        raise ValueError("All elements in N must be positive integers.")

    N_or, N_fr, N_ph, N_co = N

    # Define sampling method
    sampling = np.random.uniform if random else np.linspace

    # Generate locations
    orientation = [0] if N_or == 1 else sampling(0.0, (1 - 1 / N_or) * np.pi, N_or)
    frequency = [3] if N_fr == 1 else sampling(0.0, 9, N_fr)
    phase = (
        [np.pi / 4] if N_ph == 1 else sampling(0.0, 2 * (1 - 1 / N_ph) * np.pi, N_ph)
    )
    contrast = [1] if N_co == 1 else sampling(0.0, 1.0, N_co)

    locations = list(product(orientation, frequency, phase, contrast))

    return locations


def grating_model(
    N_neurons: tuple[int, int, int, int],
    N_stimuli: tuple[int, int, int, int],
    deltaT: Optional[float] = None,
    random_focal_points: bool = False,
    random_neurons: bool = False,
    plot_stimuli: bool = False,
) -> pd.DataFrame:
    """
    Simulate the firing of simple cells responding to images of gratings.

    Simple cells and stimuli are uniformly distributed along the parameter space.

    Parameters
    ----------
    N_neurons : Tuple[int, int, int, int]
        The number of different orientations, frequencies, phases, and contrasts for the neurons.
    N_stimuli : Tuple[int, int, int, int]
        The number of different orientations, frequencies, phases, and contrasts for the stimuli.
    deltaT : float, optional
        The time period spikes are sampled over for each stimulus. If None, return exact firing rates instead.
    random_focal_points : bool, optional
        If True, randomize the focal point for each stimulus, default is False.
    random_neurons : bool, optional
        If True, randomize the receptive field locations of the neurons, default is False.
    plot_stimuli : bool, optional
        If True, plot an image of each stimulus, default is False.

    Returns
    -------
    pd.DataFrame
        The simulated firing rate data.
    """
    Points = get_locations(N_stimuli)
    Neurons = get_locations(N_neurons, random=random_neurons)

    focal_points = (
        np.random.random([len(Points), 2]) * 2 - 1
        if random_focal_points
        else np.zeros([len(Points), 2])
    )

    rates = np.zeros([len(Points), len(Neurons)])
    iterator = trange(len(Points), desc="Simulating data points")
    for i in iterator:
        if plot_stimuli:
            grating_image(Points[i])
        for j in range(len(Neurons)):
            rates[i, j] = response(Points[i], Neurons[j], center=focal_points[i])

    data = rates if deltaT is None else poisson(rates * deltaT)
    data = pd.DataFrame(data, columns=[f"Neuron_{i}" for i in range(len(Neurons))])
    data[GRATING_PARAMS] = Points
    data.set_index(GRATING_PARAMS, inplace=True)

    if deltaT is not None:
        print("Mean spike count: " + str(data.mean().mean()))

    return data
