# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:59:03 2020

@author: Loek van Rossem
"""

import sys, os

from tqdm import trange, tqdm

import numpy as np
import array
import scipy
import scipy.io

import pandas as pd
    
import matplotlib.pyplot as plt

import sklearn
import tadasets
import gudhi

sys.path.insert(0, "./src")
from PlotData import plot_data
from AnalyzeData import analyze_data as analyze, to_dataframe, make_model, receptive_fields, PCA_reduction, glue_detection
from Gratings import plot_grating, show_feature, grating_model
from Persistence import persistence, persistence_witness
from NoiseReduction import top_noise_reduction, density_filtration, neighborhood_average, z_cutoff
from DataDecorators import multiInput


output_path = "output"


D = np.array([[2,-1,0,-1,0,0,0,0,0],[-1,3,-1,0,-1,0,0,0,0],[0,-1,2,0,0,-1,0,0,0],[-1,0,0,3,-1,0,-1,0,0],[0,-1,0,-1,4,-1,0,-1,0],[0,0,-1,0,-1,3,0,0,-1],[0,0,0,-1,0,0,2,-1,0],[0,0,0,0,-1,0,-1,3,-1],[0,0,0,0,0,-1,0,-1,2]])
def Dnorm(x):
    return np.sqrt(np.dot(x, np.dot(D, x)))

# Import data
data_path = r"D:\Research data\vanhateren_iml"
Lx, Ly = 1536, 1024
iterator = tqdm(os.listdir(data_path))
iterator.set_description("Importing images")
img = []
for filename in iterator:
    if filename.endswith(".iml") or filename.endswith(".imk"): 
        with open(os.path.join(data_path, filename), 'rb') as handle:
           s = handle.read()
        arr = array.array('H', s)
        arr.byteswap()
        img.append(np.array(arr, dtype='uint16').reshape(Ly, Lx))
        
# Sample
Npatches = 100
iterator = tqdm(img)
iterator.set_description("Sampling images")
data = pd.DataFrame()
for image in iterator:
    patches = []
    Dnorms = []
    for i in range(Npatches):
        x, y = 2 + np.random.randint(Lx - 2), 2 + np.random.randint(Ly - 2)
        sample = image[y-2:y+1, x-2:x+1]
        sample = sample.ravel()
        sample = np.log(sample)
        patches.append(sample)
        Dnorms.append(Dnorm(sample))
    # Select patches with top 20% Dnorm
    patches = pd.DataFrame(np.array(patches))
    patches["Dnorm"] = Dnorms
    patches = patches.nlargest(int(0.2 * Npatches), "Dnorm")
    patches = patches.drop(columns="Dnorm")
    data = data.append(patches, ignore_index=True)

# normalize
data = data - data.mean()
data = data.apply(lambda x : x/(Dnorm(x)), axis=1, result_type='expand')

data_backup = data

data = z_cutoff(data, 3)
# data = neighborhood_average(data,r=data.stack().std())
# data = neighborhood_average(data,r=data.stack().std())
data = density_filtration(data,k=100,fraction=0.1)

# persistence_witness(data=data, number_of_landmarks=100, max_alpha_square=5, limit_dimension=3)
# analyze(data,plot_transformation="PCA",save_path=output_path)



print("Done")
