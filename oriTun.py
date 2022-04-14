# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "./src")

from gratings import grating_model
from plotting import plot_data, plot_mean_against_index, show_feature
from persistence import persistence
from decoding import cohomological_parameterization, remove_feature
from noisereduction import PCA_reduction, z_cutoff
    


## Generate data
data = pd.DataFrame(np.genfromtxt("data/oriTun.txt",delimiter=",").T)
data["orientation"] = data.index.values.tolist()
data["orientation"] *= 30 # In experiment the value 330 seems to appear twice
data = data.set_index("orientation")

# data = grating_model(Nn=335, Np=(12,1,1,1), deltaT=None, random_neurons=True)

## Apply noise reduction
data = PCA_reduction(data, 4)
# data = z_cutoff(data,2)

## Analyze shape
persistence(data,homdim=2,coeff=2)
persistence(data,homdim=2,coeff=3)

## Decode first parameter
decoding1 = cohomological_parameterization(data, weighted=True, coeff=23)
show_feature(decoding1)
plot_mean_against_index(data,decoding1,"orientation")
plot_data(data,transformation="PCA", labels=decoding1,
          colors=["Twilight","Viridis","Twilight","Viridis","Twilight"])