# -*- coding: utf-8 -*-

#%%
## Load modules and data
import numpy as np
import pandas as pd
from ipywidgets import interactive
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "./src")

from gratings import grating_model
from plotting import plot_data, plot_mean_against_index, show_feature, plot_slider
from persistence import persistence
from decoding import cohomological_parameterization, remove_feature
from noisereduction import PCA_reduction, top_noise_reduction, z_cutoff, density_filtration

data_path = "/mnt/c/wsl/projects/top_vision/data/stringer/"
# experiment = np.load(data_path + "minnie.npy", allow_pickle=True)[()]
# experiment = np.load(data_path + "gratings_low_contrast.npy", allow_pickle=True)[()]
experiment = np.load(data_path + "gratings_drifting.npy", allow_pickle=True)[()]
# experiment = np.load(data_path + "stringer_orientations.npy", allow_pickle=True)[()]

response = experiment['sresp']
stimuli = experiment['istim']

data = pd.DataFrame(response.T, index=stimuli)

#%%
## Apply noise reduction
# data = PCA_reduction(data, 15)
# data = z_cutoff(data,2)
data = top_noise_reduction(data, n=100, speed=0.1, omega=0.3, fraction=0.1, plot_history=True)
# data = density_filtration(data,k=5,fraction=0.01)

#%%
## Analyze shape
persistence(data,homdim=1,coeff=2)
persistence(data,homdim=1,coeff=3)
plot_data(data,transformation="PCA", colors=["Twilight"])


#%%
## Decode first parameter
decoding1 = cohomological_parameterization(data, coeff=23)
images = show_feature(decoding1, Nimages=10)
interactive(lambda n : plt.imshow(images[n], "gray", vmin=-1, vmax=1), n=(0,len(images)-1,1))
plot_mean_against_index(data,decoding1,"orientation")
plot_mean_against_index(data,decoding1,"phase")
plot_data(data,transformation="PCA", labels=decoding1,
          colors=["Twilight","Viridis","Twilight","Viridis","Twilight"])


# %%
