import numpy as np
import pandas as pd


def mobius_strip_dataset(S: int = 50, T: int = 10, sigma: float = 0) -> pd.DataFrame:
    """
    Generate a Mobius strip shaped dataset.

    Parameters
    ----------
    S : int, optional
        Number of points along the length of the Mobius strip, default is 50.
    T : int, optional
        Number of points along the width of the Mobius strip, default is 10.
    sigma : float, optional
        Standard deviation of noise added to the generated points, default is 0.

    Returns
    -------
    pd.DataFrame
        Dataset in the shape of a Mobius strip.
    """
    s = np.linspace(0.0, 4 * np.pi, S)[None, :]
    t = np.linspace(-1.0, 1.0, T)[:, None]
    x = (1 + 0.5 * t * np.cos(0.5 * s)) * np.cos(s)
    y = (1 + 0.5 * t * np.cos(0.5 * s)) * np.sin(s)
    z = 0.5 * t * np.sin(0.5 * s)
    P = np.stack([x, y, z], axis=-1)
    data = pd.DataFrame(P.reshape(S * T, -1) + sigma * np.random.randn(S * T, 3))
    return data


def klein_bottle_dataset(N=15, sigma=0.05):
    """
    Generate a dataset in the shape of a Klein bottle.

    Parameters
    ----------
    N : int, optional
        Number of samples along the length and width of the bottle, default is 15.
    sigma : float, optional
        Standard deviation of the Gaussian noise added to the dataset, default is 0.05.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the generated dataset.
    """
    A = 4
    B = 1
    theta = np.linspace(0.0, 2 * np.pi, N)[None, :]
    v = np.linspace(0.0, 2 * np.pi, 2 * N)[:, None]
    x = A * (np.cos(theta / 2) * np.cos(v) - np.sin(theta / 2) * np.sin(2 * v))
    y = A * (np.sin(theta / 2) * np.cos(v) + np.cos(theta / 2) * np.sin(2 * v))
    z = B * np.cos(theta) * (1 + 0.1 * np.sin(v))
    w = B * np.sin(theta) * (1 + 0.1 * np.sin(v))
    P = np.stack([x, y, z, w], axis=-1)
    data = pd.DataFrame(P.reshape(2 * N**2, -1) + sigma * np.random.randn(2 * N**2, 4))
    return data
