"""
Some plotting functions related to grating stimuli responses
"""

import numpy as np
import pandas as pd
import math
import random

import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import imageio
from matplotlib import cm

import sklearn.manifold as manifold
from sklearn.decomposition import PCA

from tqdm import trange

from decorators import multi_input
from gratings import angular_mean, grating_image


@multi_input
def plot_data(data, transformation="None", labels=None, colors=None, save_path=None):
    """
    Plot data colored by its indices and additional provided labels

    Parameters
    ----------
    data: DataFrame(n_datapoints, n_features):
        DataFrame containing the data
    transformation: str, optional, default "None"
        The type of dimension reduction used
        Choose from "None", "PCA" or "SpectralEmbedding"
    labels: DataFrame(n_datapoints, n_labels), optional, default None
        DataFrame containing additional labels to be plotted
    colors: list of str, optional, default None
        A list containing the color scales used for each label
        When None use "Viridis" for all labels
    save_path: str, optional, default None
        When given save the figure here
    """

    # Prepare labels and colors
    indices = data.index
    plotted_labels = (
        indices.to_frame().join(labels) if labels is not None else pd.DataFrame(indices)
    )
    colors = ["Viridis"] * len(plotted_labels.columns) if colors is None else colors
    n_labels = len(plotted_labels.columns)

    # Transform data if specified
    if transformation == "PCA":
        data_transformed = PCA(n_components=3).fit_transform(data)
    elif transformation == "SpectralEmbedding":
        data_transformed = manifold.SpectralEmbedding(
            n_components=3, affinity="rbf"
        ).fit_transform(data)
    elif transformation == "None":
        data_transformed = data
    else:
        raise ValueError(
            "Invalid plot transformation. Choose from 'None', 'PCA', or 'SpectralEmbedding'."
        )

    # Plot the data
    fig = make_subplots(
        rows=2,
        cols=math.ceil(n_labels / 2),
        specs=[[{"type": "scene"}] * math.ceil(n_labels / 2)] * 2,
    )
    for i, label in enumerate(plotted_labels.columns):
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                name=label,
                x=data_transformed[:, 0],
                y=data_transformed[:, 1],
                z=data_transformed[:, 2],
                text=plotted_labels[label],
                hoverinfo=["x", "text"],
                marker=dict(
                    color=plotted_labels[label],
                    size=5,
                    sizemode="diameter",
                    colorscale=colors[i],
                ),
            ),
            row=i % 2 + 1,
            col=math.floor(i / 2) + 1,
        )
    fig.update_layout(height=800, width=600 * math.ceil(n_labels / 2), title_text="")

    # Show or save the plot
    if save_path is None:
        plot(fig)
        fig.show()
    else:
        file_path = (
            save_path
            + f"_plot_{transformation if transformation != 'None' else ''}.html"
        )
        fig.write_html(file_path)


def plot_connections(
    data_points, connections, threshold=0.1, opacity=0.1, save_path=None
):
    """
    Plot connections between data points in a 3D space.

    Parameters
    ----------
    data_points : numpy.ndarray
        3D array containing the coordinates of data points.
    connections : numpy.ndarray
        2D array containing the connection strengths between data points.
    threshold : float, optional
        Threshold value for considering connections, defaults to 0.1.
    opacity : float, optional
        Opacity level of the lines, defaults to 0.1.
    save_path : str, optional
        Path to save the plot, defaults to None (showing the plot interactively).
    """
    # Draw a square
    x = data_points[:, 0]
    y = data_points[:, 1]
    z = data_points[:, 2]
    N = len(x)

    # The start and end point for each line
    pairs = [(i, j) for i, j in np.ndindex((N, N)) if connections[i, j] > threshold]

    trace1 = go.Scatter3d(x=x, y=y, z=z, mode="markers", name="markers")

    # Create the coordinate list for the lines
    x_lines, y_lines, z_lines = [], [], []
    for p in pairs:
        for i in range(2):
            x_lines.append(x[p[i]])
            y_lines.append(y[p[i]])
            z_lines.append(z[p[i]])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)

    trace2 = go.Scatter3d(
        x=x_lines, y=y_lines, z=z_lines, opacity=opacity, mode="lines", name="lines"
    )

    fig = go.Figure(data=[trace1, trace2])

    if save_path is not None:
        path = save_path
        path += "/plot_glue.html"
        fig.write_html(path)
    else:
        plot(fig)
    return


@multi_input
def plot_mean_against_index(data, value, index, circ=True, save_path=None):
    """
    Plot the mean value against an index.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the data.
    value : pandas.DataFrame
        The value we average over.
    index : str
        The name of the index we plot against.
    circ : bool, optional
        Whether or not the index is angular, defaults to True.
    save_path : str, optional
        Path to save the plot, defaults to None (showing the plot interactively).
    """
    unique_index = data.reset_index()[index].unique()
    if len(unique_index) > 1:
        if circ:
            means = value.groupby([index]).agg(lambda X: angular_mean(X, period=1))
        else:
            means = value.groupby([index]).mean()

        plt.figure(figsize=(3, 3))
        plt.scatter(means.index, means)
        if circ:
            plt.ylim(0, 1)
        plt.xlabel(index)
        plt.ylabel(f"mean {value.columns[0]}")

        if save_path is not None:
            plt.savefig(f"{save_path}/plot_mean_against_index.png")
            plt.close()
        else:
            plt.show()

    return


@multi_input
def show_feature(
    decoding,
    Nimages=10,
    Npixels=100,
    normalized=True,
    intervals="equal_images",
    save_path=None,
):
    """
    Show how the gratings depend on a decoded parameter

    Shows the average grating for different values of the decoding and plot the
    grating parameters as a function of the decoding

    Parameters
    ----------
    decoding : dataframe(n_datapoints)
        A dataframe containing the decoded value for each data point
        labeled by indices "orientation, "frequency", "phase" and "contrast"
    Nimages : int, optional, default 10
        Number of different images shown
    Npixels: int, optional, default 100
        The number of pixels in each direction
    normalized : bool, optional, default True
       If true normalize the average images
    intervals : str, optional, default "equal_images"
        How the images are binned together
        - When "equal_images" average such that each image is an average of
          an equal number of images
        - When "equal_decoding" average such that each image is averaged from
          images within an equal fraction of the decoding
    save_path: str, optional, default None
        If provided, save a gif here
    """
    try:
        orientation = decoding.reset_index()["orientation"]
    except KeyError:
        orientation = pd.Series(np.zeros(len(decoding)))
    try:
        contrast = decoding.reset_index()["contrast"]
    except KeyError:
        contrast = pd.Series(np.ones(len(decoding)))
    try:
        frequency = decoding.reset_index()["frequency"]
    except KeyError:
        frequency = pd.Series(0.1 * np.ones(len(decoding)))
    try:
        phase = decoding.reset_index()["phase"]
    except KeyError:
        phase = pd.Series(np.zeros(len(decoding)))

    decoding = decoding.to_numpy().ravel()
    N = decoding.shape[0]

    # Convert grating labels
    if np.max(phase) > 2 * np.pi:
        orientation = np.radians(orientation)
        frequency *= 30
        phase = np.radians(phase)

    # Assign images to intervals
    interval_labels = np.zeros(N, dtype=int)
    count = np.zeros(Nimages)

    if intervals == "equal_decoding":
        intervals = np.linspace(np.min(decoding), np.max(decoding), Nimages + 1)
        for i in range(Nimages):
            interval_indices = np.where(
                (intervals[i] <= decoding) & (decoding <= intervals[i + 1])
            )[0]
            interval_labels[interval_indices] = i
            count[i] = len(interval_indices)
    elif intervals == "equal_images":
        sorted_indices = np.argsort(decoding)
        interval_labels[sorted_indices] = np.floor(
            np.linspace(0, Nimages - 0.01, N)
        ).astype(int)
        for i in range(Nimages):
            count[i] = np.sum(interval_labels == i)
    else:
        raise ValueError("Invalid intervals type")

    # Average over grating images
    av_images = np.zeros([Nimages, Npixels, Npixels])
    iterator = trange(Nimages, desc="Averaging images", position=0, leave=True)
    for i in iterator:
        if count[i] != 0:
            indices = np.where(interval_labels == i)[0]
            for j in indices:
                pars = np.array([orientation[j], frequency[j], phase[j], contrast[j]])
                av_images[i] += grating_image(pars, N=Npixels, plot=False)
            av_images[i] /= count[i]

    print("Number of images averaged over:")
    print(count)

    if normalized:
        av_images = av_images / np.max(np.abs(av_images))

    # Show averages and make gif
    fig, axs = plt.subplots(1, Nimages, figsize=(10 * Nimages, 10))
    for i in range(Nimages):
        axs[i].imshow(av_images[i], cmap="Greys")
        axs[i].axis("off")
        if save_path is not None:
            plt.savefig(f"{save_path}_image{i + 1}.png")

    if save_path is not None:
        frames = (255 * 0.5 * (1 + av_images)).astype(np.uint8)
        imageio.mimsave(f"{save_path}.gif", frames)

    plt.show()

    return av_images


def receptive_fields(data, feature_x, feature_y, N=100):
    """
    Plot the 2D receptive fields of neurons responding to two features.

    Parameters
    ----------
    data : pd.DataFrame(n_neurons)
        The firing rate at each data point.
    feature_x : ndarray(n_x)
        The first feature to plot against.
    feature_y : ndarray(n_y)
        The second feature to plot against.
    N : int, optional, default 100
        The number of pixels in each direction.
    """
    n_neurons = len(data.columns)
    fig, axs = plt.subplots(1, n_neurons, figsize=(10 * n_neurons, 10))
    for ax, neuron in zip(axs, data.columns):
        x = np.linspace(0, 1, N)
        xv, yv = np.meshgrid(x, x)

        values = data[neuron].values - np.mean(data.values)
        values /= np.ptp(values)
        colors = cm.rainbow(values)
        ax.scatter(feature_x, feature_y, color=colors, s=3000)
        ax.axis("off")
    plt.show()



def plot_slider(data_list):
    """
    Create a plot with a slider to vary which dataset is shown.

    Parameters
    ----------
    data_list: list
        List containing DataFrames.
    """
    # Create figure
    fig = go.Figure()

    # Add traces, one for each DataFrame in data_list
    for i, df in enumerate(data_list):
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                visible=(i == 0),  # Show the first dataset initially
                opacity=0.5,
                name=f"Dataset {i+1}",
                x=df[0],
                y=df[1],
                z=df[3],
                marker=dict(size=3),
            )
        )

    # Create slider steps
    steps = []
    for i in range(len(data_list)):
        step = dict(
            method="update",
            args=[
                {
                    "visible": [(j == i) for j in range(len(data_list))]
                },  # Show only the selected dataset
                {
                    "title": f"Dataset {i+1}"
                },  # Update title with the selected dataset number
            ],
        )
        steps.append(step)

    # Add slider
    sliders = [
        dict(
            active=0,  # Start with the first dataset active
            currentvalue={"prefix": "Dataset: "},
            pad={"t": 50},
            steps=steps,
        )
    ]

    fig.update_layout(sliders=sliders)
    fig.show()
