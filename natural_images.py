# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:59:03 2020

@author: Loek van Rossem
"""

import sys

import numpy as np
import scipy
import scipy.io

import pandas as pd
    
import matplotlib.pyplot as plt

import sklearn
import tadasets

sys.path.insert(0, "./src")
from PlotData import plot_data
from AnalyzeData import analyze_data as analyze, to_dataframe, make_model, receptive_fields, PCA_reduction, glue_detection
from Gratings import plot_grating, show_feature, grating_model
from Persistence import persistence
from NoiseReduction import noise_reduction
from DataDecorators import multiInput

output_path = "output"


data_list = {}
data_list['pvc8_1'] = scipy.io.loadmat(r"data\crncs_pvc8\01.mat")
data_list['pvc8_2'] = scipy.io.loadmat(r"data\crncs_pvc8\02.mat")
data_list['pvc8_3'] = scipy.io.loadmat(r"data\crncs_pvc8\03.mat")
data_list['pvc8_4'] = scipy.io.loadmat(r"data\crncs_pvc8\04.mat")
data_list['pvc8_5'] = scipy.io.loadmat(r"data\crncs_pvc8\05.mat")
data_list['pvc8_6'] = scipy.io.loadmat(r"data\crncs_pvc8\06.mat")
data_list['pvc8_7'] = scipy.io.loadmat(r"data\crncs_pvc8\07.mat")
data_list['pvc8_8'] = scipy.io.loadmat(r"data\crncs_pvc8\08.mat")
data_list['pvc8_9'] = scipy.io.loadmat(r"data\crncs_pvc8\09.mat")
data_list['pvc8_10'] = scipy.io.loadmat(r"data\crncs_pvc8\10.mat")

@multiInput
def process_data(mat):
    images = mat['images'][0]   # First 540 natural
    spikes = mat['resp_train']
    
    rates = np.average(spikes, axis=2) # Average over trails
    data = np.average(rates, axis=2) # Average over time
    # data = rates.reshape(rates.shape[0], rates.shape[1] * rates.shape[2])
    data = data.transpose()
    data = pd.DataFrame(data)
    # data = data[(data.T != 0).any()]    # Remove empty rows
    
    # data = data[540:]
    data = data[:540]
    
    return data
    
data_list = process_data(data_list)

# plt.imshow(images[0])


## Analyze data

# analyze(data,homdim=1,plot_transformation="PCA",save_path=output_path)
# data = PCA_reduction(data, 5)
# data = noise_reduction(data, n=100, omega=0.5, fraction=0.9)

analyze(data_list,coeff=2,homdim=2,plot_transformation="PCA",save_path=output_path)
# reduced_data, decoding1 = analyze(data,coeff=23,decode_feature=1,shift=0,save_path=output_path,plot_transformation="PCA", decode="Cohomology", Nimages=10)
# analyze(reduced_data, homdim=2, plot_transformation="PCA", save_path=output_path)
# even_more_reduced_data, decoding2 = analyze(reduced_data,coeff=23,decode_feature=1,save_path=output_path,plot_transformation="PCA", decode="Cohomology", Nimages=10)

# _, decoding1 = analyze(data,homdim=1,coeff=23,decode_feature=1,save_path=output_path,plot_transformation="PCA", decode="PCA", Nimages=10, dim_estimation=False)
# _, decoding2 = analyze(data,homdim=1,coeff=23,decode_feature=2,save_path=output_path,plot_transformation="PCA", decode="Cohomology", Nimages=10, dim_estimation=False)
    
# glue_detection(data,threshold=0.5,save_path=output_path)

# receptive_fields(data_backup, decoding1, decoding2) 
# orientation = list(data.values())[0].reset_index()['orientation']/np.pi
# frequency = list(data.values())[0].reset_index()['frequency']/10
# phase = list(data.values())[0].reset_index()['phase']/(2*np.pi)
# orientation, frequency, phase = orientation/np.max(orientation), frequency/np.max(frequency), phase/np.max(phase)
# receptive_fields(data, orientation, frequency)


print("Done")
