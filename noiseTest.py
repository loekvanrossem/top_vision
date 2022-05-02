# -*- coding: utf-8 -*-

#%%
import numpy as np
import pandas as pd
from ipywidgets import interactive
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "./src")

from gratings import grating_model
from plotting import plot_data, plot_mean_against_index, show_feature, receptive_fields
from persistence import persistence
from decoding import cohomological_parameterization, remove_feature
from noisereduction import PCA_reduction, top_noise_reduction, z_cutoff



## Generate data
data = grating_model(Nn=5, Np=(18,1,18,1), deltaT=1000, random_neurons=False)

#%%
## Apply noise reduction
# data = PCA_reduction(data, 5)
# data = z_cutoff(data,2)

data = top_noise_reduction(data, n=100, omega=0.3, fraction=0.5, plot_history=True)


#%%
## Analyze shape
persistence(data,homdim=2,coeff=2)
persistence(data,homdim=2,coeff=3)

## Decode first parameter
decoding1 = cohomological_parameterization(data, coeff=23)
images = show_feature(decoding1, Nimages=10)
interactive(lambda n : plt.imshow(images[n], "gray", vmin=-1, vmax=1), n=(0,len(images)-1,1))
plot_mean_against_index(data,decoding1,"orientation")
plot_mean_against_index(data,decoding1,"phase")
# plot_data(data,transformation="PCA", labels=decoding1,
#           colors=["Twilight","Viridis","Twilight","Viridis","Twilight"])

## Decode second parameter
# reduced_data = remove_feature(data, decoding1, cut_amplitude=0.5)
# decoding2 = cohomological_parameterization(reduced_data, coeff=23)
# show_feature(decoding2)
# plot_mean_against_index(data,decoding2,"orientation")
# plot_mean_against_index(data,decoding2,"phase")
# plot_data(data,transformation="PCA", labels=decoding2,
#           colors=["Twilight","Viridis","Twilight","Viridis","Twilight"])
# %%
## Plot tuning
receptive_fields(data, data.reset_index()["orientation"]/np.pi, data.reset_index()["phase"]/2*np.pi)

# %%
