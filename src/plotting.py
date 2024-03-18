# -*- coding: utf-8 -*-
"""
Some plotting functions related to grating stimuli responses
"""
import numpy as np
import pandas as pd
import math
import random
import scipy

import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import imageio
from matplotlib import cm
from ipywidgets import interactive

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
    data: dataframe(n_datapoints, n_features):
        Dataframe containing the data
    transformation: str, optional, default "None"
        The type of dimension reduction used
        Choose from "None", PCA" or "SpectralEmbedding"
    labels: dataframe(n_datapoints, n_labels), optional, default None
        Dataframe containing additional labels to be plotted
    colors: list of str, optional, default None
        A list containing the color scales used for each label
        When None use "Viridis" for all labels
    save_path: str, optional, default None
        When given save the figure here

    Raises
    ------
    ValueError
        When an invalid value for "transformation" is given
    """
    # Set labels
    indices = data.index
    plotted_labels = indices.to_frame()
    if labels is not None:
        plotted_labels = plotted_labels.join(labels)
    if colors is None:
        colors = ["Viridis"] * len(plotted_labels.columns)
    Nlabels = len(plotted_labels.columns)

    # Transform data to lower dimension
    if transformation == "PCA":
        pca = PCA(n_components=3)
        pca.fit(data)
        data = pca.transform(data)
    elif transformation == "SpectralEmbedding":
        embedding = manifold.SpectralEmbedding(n_components=3, affinity="rbf")
        data = embedding.fit_transform(data)
    elif transformation == "None":
        pass
    else:
        raise ValueError("Invalid plot transformation")

    # Plot
    data = pd.DataFrame(data)
    data.index = indices
    fig = go.Figure()
    fig = make_subplots(
        rows=2,
        cols=math.ceil(Nlabels / 2),
        specs=[[{"type": "scene"}] * math.ceil(Nlabels / 2)] * 2,
    )
    for i, label in enumerate(plotted_labels):
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                name=label,
                x=data[0],
                y=data[1],
                z=data[2],
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
    fig.update_layout(height=800, width=600 * math.ceil(Nlabels / 2), title_text="")
    if save_path is None:
        plot(fig)
        fig.show()
    else:
        path = save_path
        path += "plot"
        if transformation != "None":
            path += "_" + transformation
        path += ".html"
        fig.write_html(path)
    return


def plot_connections(
    data_points, connections, threshold=0.1, opacity=0.1, save_path=None
):
    # draw a square
    x = data_points[:, 0]
    y = data_points[:, 1]
    z = data_points[:, 2]
    N = len(x)

    # the start and end point for each line
    pairs = [(i, j) for i, j in np.ndindex((N, N)) if connections[i, j] > threshold]

    trace1 = go.Scatter3d(x=x, y=y, z=z, mode="markers", name="markers")

    x_lines = list()
    y_lines = list()
    z_lines = list()

    # create the coordinate list for the lines
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
    Plot the mean value against an index

    Parameters
    ----------
    data: dataframe(n_datapoints, n_features):
        Dataframe containing the data
    value : dataframe(n_datapoints)
        The value we average over
    index : str
        The name of the index we plot against
    circ : bool, optional, default True
        Whether or not the index is angular
    """
    unique_index = data.reset_index()[index].unique()
    if len(unique_index) > 1:
        if circ:
            means = value.groupby([index]).agg(lambda X: angular_mean(X, period=1))
        else:
            means = value.groupby([index]).mean()
        plt.scatter(means.index, means)
        if circ:
            plt.ylim(0, 1)
        plt.xlabel(index)
        plt.ylabel(f"mean {value.columns[0]}")
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
        If given, save a gif here

    Raises
    ------
    ValueError
        When an invalid value for "intervals" is given

    Warns
    -----
        When some of the bins are empty
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
    if max(phase) > 2 * np.pi:
        orientation = orientation * np.pi / 180
        frequency = frequency * 30
        phase = phase * np.pi / 180

    # Assign images to intervals
    interval_labels = np.zeros(N, dtype=int)
    count = np.zeros(Nimages)
    if intervals == "equal_decoding":
        intervals = np.linspace(min(decoding), max(decoding), Nimages + 1)
        for i in range(Nimages):
            for j in range(N):
                if (intervals[i] <= decoding[j]) and (decoding[j] <= intervals[i + 1]):
                    interval_labels[j] = i
                    count[i] += 1
    elif intervals == "equal_images":
        interval_labels = (np.floor(np.linspace(0, Nimages - 0.01, N))).astype(int)
        interval_labels = interval_labels[np.argsort(decoding)]
        for i in range(Nimages):
            count[i] = (interval_labels[interval_labels == i]).shape[0]
    else:
        raise ValueError("Invalid intervals type")
        return

    # Average over grating images
    av_images = np.zeros([Nimages, Npixels, Npixels])
    grouped_indices = [[] for _ in range(Nimages)]
    iterator = trange(0, N, position=0, leave=True)
    iterator.set_description("Averaging images")
    for j in iterator:
        pars = np.array([orientation[j], frequency[j], phase[j], contrast[j]])
        grouped_indices[interval_labels[j]].append(pars)
        if random.choice([True, False]):
            av_images[interval_labels[j]] += grating_image(pars, N=Npixels, plot=False)
    for i in range(Nimages):
        if count[i] != 0:
            av_images[i] = av_images[i] / count[i]

    print("Number of images averaged over:")
    print(count)

    if normalized:
        av_images = av_images / np.max(np.abs(av_images))

    # Show averages and make gif
    for i in range(Nimages):
        if not math.isnan(av_images[i, 0, 0]):
            if save_path is not None:
                plt.savefig(save_path + "_image" + str(i + 1) + ".png")
    frames = (255 * 0.5 * (1 + av_images)).astype(np.uint8)
    if save_path is not None:
        imageio.mimsave(save_path + ".gif", frames)
    fig, axs = plt.subplots(1, len(frames), figsize=(10, 10))
    for ax, frame in zip(axs, frames):
        ax.imshow(frame, cmap="Greys")
        ax.axis("off")
    plt.show()

    # Plot the grating parameters against the decoding
    # try:
    #     ori_fun = lambda x: angular_mean(np.array(x)[:, 0], period=np.pi, axis=0)
    #     freq_fun = lambda x: np.mean(np.array(x)[:, 1], axis=0)
    #     phase_fun = lambda x: angular_mean(np.array(x)[:, 2], period=2 * np.pi, axis=0)
    #     contrast_fun = lambda x: np.mean(np.array(x)[:, 3], axis=0)
    #     av_orientation = np.array(list(map(ori_fun, grouped_indices)))
    #     av_frequency = np.array(list(map(freq_fun, grouped_indices)))
    #     av_phase = np.array(list(map(phase_fun, grouped_indices)))
    #     av_contrast = np.array(list(map(contrast_fun, grouped_indices)))
    # except IndexError:
    #     print("Error: some bins are empty; choose a smaller bin count.")
    # else:
    #     if len(set(orientation)) > 1:
    #         plt.scatter(np.arange(Nimages), av_orientation)
    #         plt.xlabel("Decoding")
    #         plt.ylabel("Orientation")
    #         plt.show()
    #     if len(set(frequency)) > 1:
    #         plt.scatter(np.arange(Nimages), av_frequency)
    #         plt.xlabel("Decoding")
    #         plt.ylabel("Frequency")
    #         plt.show()
    #     if len(set(phase)) > 1:
    #         plt.scatter(np.arange(Nimages), av_phase)
    #         plt.xlabel("Decoding")
    #         plt.ylabel("Phase")
    #         plt.show()
    #     if len(set(contrast)) > 1:
    #         plt.scatter(np.arange(Nimages), av_contrast)
    #         plt.xlabel("Decoding")
    #         plt.ylabel("Contrast")
    #         plt.show()
    return av_images


def receptive_fields(data, feature_x, feature_y, N=100):
    """
    Plot the 2d receptive fields of neurons responding to two features

    Parameters
    ----------
    data : ndarray(n_x, n_y)
        The firing rate at each datapoint.
    feature_x : ndarray(n_x)
        The first feature to plot against.
    feature_y : ndarray(n_y)
        The second feature to plot against.
    N : int, optional, default 100
        The number of pixels in each direction.
    """
    for neuron in data:
        x = np.linspace(0, 1, N)
        xv, yv = np.meshgrid(x, x)

        values = data[neuron] - np.mean(np.mean(data))
        values = values / np.ptp(values)
        colors = cm.rainbow(values)
        plt.scatter(feature_x, feature_y, color=colors)
        plt.show()

        interpolated_data = scipy.interpolate.griddata(
            pd.concat([feature_x, feature_y], axis=1), data[neuron], (xv, yv)
        )
        plt.imshow(interpolated_data, origin="lower")
        plt.show()


def plot_slider(data_list):
    """
    Create a plot with a slider to vary which data set is shown

    Parameters
    ----------
    data_list: list
        list containing dataframes
    """
    # Create figure
    fig = go.Figure()

    # Add traces, one for each slider step
    for df in data_list:
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                visible=False,
                opacity=0.5,
                name="𝜈 = ",
                x=df[0],
                y=df[1],
                z=df[3],
                marker=dict(
                    size=3,
                ),
            )
        )

    # Start with trace visible
    fig.data[-1].visible = True

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
                {"title": "Slider switched to step: " + str(i)},
            ],
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)

    sliders = [dict(active=10, currentvalue={"prefix": ""}, pad={"t": 50}, steps=steps)]

    fig.update_layout(sliders=sliders)

    fig.show()
