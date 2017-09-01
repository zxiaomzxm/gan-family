# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 13:38:16 2017

@author: zhaoxm
"""

import tensorflow as tf
import tensorflow.contrib.distributions as ds
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
import itertools
import seaborn as sb

def gaussian_mixture_circle(batchsize, num_cluster=8, scale=1, std=1):
	rand_indices = np.random.randint(0, num_cluster, size=batchsize)
	base_angle = np.pi * 2 / num_cluster
	angle = rand_indices * base_angle - np.pi / 2
	mean = np.zeros((batchsize, 2), dtype=np.float32)
	mean[:, 0] = np.cos(angle) * scale
	mean[:, 1] = np.sin(angle) * scale
	return np.random.normal(mean, std**2, (batchsize, 2)).astype(np.float32)

def gaussian_mixture_double_circle(batchsize, num_cluster=8, scale=1, std=1):
	rand_indices = np.random.randint(0, num_cluster, size=batchsize)
	base_angle = np.pi * 2 / num_cluster
	angle = rand_indices * base_angle - np.pi / 2
	mean = np.zeros((batchsize, 2), dtype=np.float32)
	mean[:, 0] = np.cos(angle) * scale
	mean[:, 1] = np.sin(angle) * scale
	# Doubles the scale in case of even number
	even_indices = np.argwhere(rand_indices % 2 == 0)
	mean[even_indices] /= 2
	return np.random.normal(mean, std**2, (batchsize, 2)).astype(np.float32)
       
def create_mixgaussian2D(num_components=8, std=0.05):
    cat = ds.Categorical(tf.zeros(num_components, dtype=tf.float32))
#    mus = np.array([np.array([i, j]) for i, j in itertools.product(np.linspace(-1, 1, 5),
#                                                           np.linspace(-1, 1, 5))],dtype=np.float32)
    mus = np.array([(np.cos(theta), np.sin(theta)) for theta in np.linspace(0, 2*np.pi, num_components+1)],dtype=np.float32)
#    mus = (mus + 2) / 4.
                                                           
    sigmas = [np.array([std, std]).astype(np.float32) for i in range(num_components)]
    components = list((ds.MultivariateNormalDiag(mu, sigma) 
                       for (mu, sigma) in zip(mus, sigmas)))
    data = ds.Mixture(cat, components)
    return data
    
    
def kdeplot(samples,scale=100, window_size=5):
    resolution = 1. / scale
    density_estimation = np.zeros((scale, scale))
    for x, y in samples:
        if 0 < x < 1 and 0 < y < 1:
            density_estimation[int((1 - y) / resolution)][int(x / resolution)] += 1
    density_estimation = filters.gaussian_filter(density_estimation, window_size)
    plt.imshow(density_estimation, cmap='Blues')
        
        
if __name__ == '__main__':
#    data = create_mixgaussian2D(num_components=8)
#    sess = tf.Session()
#    samples = sess.run(data.sample(500))
    samples = gaussian_mixture_circle(500, num_cluster=8, scale=1, std=0.01)
#    samples = gaussian_mixture_double_circle(500, num_cluster=16, scale=1, std=0.01)
#    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
#    ax[0].plot(samples[:,0], samples[:,1], '.')
    sb.kdeplot(samples, n_levels=100, shade=True, cmap='Blues')
    

    
    
    
