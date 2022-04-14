# -*- coding: utf-8 -*-
"""
"""
import os

import numpy as np
import scipy
import scipy.spatial.distance as sd
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm

import sklearn.manifold as manifold
from sklearn.decomposition import PCA
import skdim

from noisereduction import top_noise_reduction
from dimension import estimate_dimension
from persistence import persistence
from decoding import decode as cohomological_decoding, remove_feature
from gratings import grating_model
from plotting import plot_data, show_feature, plot_connections
from decorators import multi_input


def to_dataframe(data):
    df = pd.DataFrame(data)
    index = pd.DataFrame({'orientation': data[:,0], 'contrast': data[:,1], 'frequency': data[:,2], 'phase': data[:,3]})
    df = pd.DataFrame(data[:,4:])
    df = pd.concat([index,df], axis=1)
    df = df.set_index(['orientation', 'contrast', 'frequency', 'phase'])
    return df


def correlator(x, y):
    return np.abs(np.corrcoef(x, y)[0,1])
    

def to_cell_space(spikes):
    data = np.zeros([spikes.shape[1], spikes.shape[1]])
    for index_1, neuron_1 in enumerate(data):
        for index_2, neuron_2 in enumerate(data):
            data[index_1, index_2] = np.exp(-correlator(spikes[index_1], spikes[index_2]))
    return data


def make_model(Nn, Np_or=1,Np_fr=1,Np_ph=1,Np_co=1,plot_stimuli=False,
               deltaT=None,random_focal_points=False,average=1):
    model_name = 'model_Nn' + str(Nn)
    if Np_or != 1:
        model_name += '_Npor' + str(Np_or)
    if Np_fr != 1:
        model_name += '_Npfr' + str(Np_fr)
    if Np_ph != 1:
        model_name += '_Npph' + str(Np_ph)
    if Np_co != 1:
        model_name += '_Npco' + str(Np_co)
    # if deltaT is not None:
    #     model_name += '_deltaT' + str(deltaT)
        
    data_model = {}
    data_model[model_name]=grating_model(Nn=Nn, Np=(Np_or,Np_fr,Np_ph,Np_co),
                                         receptive_field_sigma = 5,
                                         plot_stimuli=plot_stimuli,
                                         deltaT=deltaT,
                                         random_focal_points=random_focal_points,
                                         average=average)
    if deltaT is not None:
        print("Mean spike count: " + str(data_model[model_name].mean().mean()))
    return data_model

def analyze_data(data_list,orfrphco=[True,True,True,True],PCA_dim=None,
                 cell_space=False,noise_reduction_fraction=None,
                 noise_reduction_steps=100,z_cutoff=None,coeff=None,
                 homdim=1,decode=None,decode_feature=1,shift=0,Nsubsamples=0,
                 Nimages=None,dim_estimation=False,save_path=None,
                 plot_transformation="None"):
    if not (isinstance(data_list, list) or isinstance(data_list, dict)):
        data_list = [data_list]
        
    use_orientation = orfrphco[0]
    use_frequency = orfrphco[1]
    use_phase = orfrphco[2]
    use_contrast = orfrphco[3]

    if decode == "CohomologyWeighted":
        decode = "Cohomology"
        weighted_decoding=True
    else:
        weighted_decoding=False
    
    reduced_data = {}
    name_count = 0 
    
    for data in data_list:
        
        ## Set name
        name=None
        if isinstance(data_list, dict):
            name = data
            data = data_list[name]
        else:
            name_count += 1
            name = 'data' + str(name_count)
        
        if not (use_orientation and use_frequency and use_phase and use_contrast):
            if use_orientation:
                name += '_orientation'
            if use_frequency:
                name += '_frequency'
            if use_phase:
                name += '_phase'
            if use_contrast:
                name += '_contrast'
        
        ## Set save path
        save_path_data=None
        if save_path is not None:
            save_path_data = save_path + '/' + name
            try:
                os.makedirs(save_path_data)
            except FileExistsError:
                pass
            save_path_data += '/'
            if decode is not None:
                try:
                    os.makedirs(save_path_data + 'feature' + str(decode_feature) + '_Z' + str(coeff))
                except FileExistsError:
                    pass
        
        ## Limit to certain labels
        if not use_frequency:
            data = data.query('frequency == 0.332966')
        if not use_orientation:
            data = data.query('orientation == 0.0')
        if not use_phase:
            data = data.query('phase == 0.0')
        if not use_contrast:
            data = data.query('contrast == 1.0')
        
        ## Transform to cell space
        # warning: some of the next processing steps might not make sense in cell space
        if cell_space:
            distance_matrix=True
            data = to_cell_space(data)
        else:
            distance_matrix=False
        
        ## Reduce dimension
        if PCA_dim is not None:
            data = PCA_reduction(data, PCA_dim)
        
        ## Reduce noise
        if noise_reduction_fraction is not None:
            data = top_noise_reduction(data, n=noise_reduction_steps, omega=0.6, fraction = noise_reduction_fraction)
        
        # Remove outliers
        if z_cutoff is not None:
            z = np.abs(scipy.stats.zscore(np.sqrt(np.square(data).sum(axis=1))))
            z_labels = pd.DataFrame(z, columns=["Z-score"])
            z_labels = z_labels.set_index(data.index)
            plot_data(data, transformation=plot_transformation, labels=z_labels,
                      colors=["Twilight","Viridis","Viridis","Twilight","Viridis"],
                      save_path = save_path_data)
            data = data[(z < z_cutoff)]
            print(data)
        
        ## Estimate dimension
        if dim_estimation:
            estimate_dimension(data, max_size=data.to_numpy().max(), test_size=30, Nsteps = 50, fraction=0.9)
            danco = skdim.id.DANCo().fit(data)
            print(danco.dimension_)
            
            
        if decode is not None:
            if decode == "Cohomology":
                decoding = cohomological_decoding(data, coeff=[coeff,23][coeff is None], weighted=weighted_decoding, cocycle_number=decode_feature, name=name, save_path=save_path_data)
            elif decode == "SpectralEmbedding":
                decoding = manifold.SpectralEmbedding(n_components=1, affinity='rbf').fit_transform(data)
            elif decode == "PCA":
                pca = PCA(n_components=decode_feature)
                pca.fit(data)
                decoding = pca.transform(data)[:,decode_feature-1]
            else:
                raise Exception("Error: invalid decode parameter")
            decoding = pd.DataFrame(decoding, columns=["decoding"])
            decoding = decoding.set_index(data.index)
            
            
            plot_data(data, transformation=plot_transformation, labels=decoding, colors=["Twilight","Viridis","Viridis","Twilight",["Viridis","Twilight"][decode == "Cohomology"]], save_path=save_path_data)
            if Nimages is not None:
                show_feature(decoding, Nimages = Nimages, Npixels = 100, intervals="equal_decoding", save_path=save_path_data + '/feature' + str(decode_feature) + '_Z' + str(coeff) + '/feature' + str(decode_feature))
            
            reduced_data[name + '_feature' + str(decode_feature) + 'removed'] = remove_feature(data, decoding, shift=shift)
        else:
            plot_data(data, transformation=plot_transformation, colors=["Twilight","Viridis","Viridis","Twilight"], save_path=save_path_data)
            if coeff is not None:
                persistence(data, homdim=homdim, coeff=coeff, Nsubsamples=Nsubsamples, name=name, save_path=save_path_data, distance_matrix=distance_matrix)

    if decode is not None:
        return reduced_data, decoding
    else:
        return
    
    
def receptive_fields(data, decodingx, decodingy):
    N=100
    for neuron in data:
        x = np.linspace(0, 1, N)
        xv, yv = np.meshgrid(x, x)

        colors = cm.rainbow(data[neuron])
        plt.scatter(decodingx, decodingy, color=colors)
        plt.show()
        
        interpolated_data = scipy.interpolate.griddata(pd.concat([decodingx, decodingy], axis=1), data[neuron], (xv,yv))
        plt.imshow(interpolated_data, origin='lower')
        plt.show()
            
        
# Plot points that are glued together
@multi_input
def glue_detection(dataframe, **kwargs):
    labels = dataframe.index.to_frame().to_numpy()
    labels = labels[:,0:3]    # remove contrast
    labels_normalized = labels/np.max(labels, axis=0)
    data = dataframe.to_numpy()
    distances_labels = sd.squareform((sd.pdist(labels_normalized)))
    distances_data = sd.squareform((sd.pdist(data)))
    glue = distances_labels / distances_data
    glue = np.nan_to_num(glue)
    glue = glue/np.amax(glue)
    print(glue)
    plot_connections(labels, glue, **kwargs)
    return

