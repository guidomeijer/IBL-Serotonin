#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 15:35:50 2022
By: Guido Meijer
"""

import numpy as np
import seaborn as sns
import ssm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from serotonin_functions import figure_style, get_neuron_qc, load_passive_opto_times
from brainbox.io.one import SpikeSortingLoader
from matplotlib import cm
from matplotlib.colors import ListedColormap
from brainbox.singlecell import calculate_peths
from ibllib.atlas import AllenAtlas
from one.api import ONE
one = ONE()
ba = AllenAtlas()

K = 2    # number of discrete states
D = 10   # PCA dimensions to use
N_FRAMES = 2000  # nr of frames to use
FRAME_RATE = 60 # sampling frequency
BIN_SIZE = 0.2
SMOOTHING = 0.1
PID = '605aaf5d-9a95-4c2d-8cb9-dccf596c6452'

# Load in opto times
opto_times, _ = load_passive_opto_times(one.pid2eid(PID)[0], one=one)

# Load in neural data
sl = SpikeSortingLoader(pid=PID, one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)

# Filter neurons that pass QC
qc_metrics = get_neuron_qc(PID, one=one, ba=ba)
clusters_pass = np.where(qc_metrics['label'] == 1)[0]
spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]

# Get smoothed firing rates
peth, _ = calculate_peths(spikes.times, spikes.clusters, np.unique(spikes.clusters),
                          [opto_times[0]-1], pre_time=0, post_time=(opto_times[-1] - opto_times[0])+1,
                          bin_size=BIN_SIZE, smoothing=SMOOTHING)
tscale = peth['tscale'] + (opto_times[0]-1)

# Do PCA
pca = PCA(n_components=D)
ss = StandardScaler(with_mean=True, with_std=True)
pop_vector_norm = ss.fit_transform(peth['means'].T)
pca_proj = pca.fit_transform(pop_vector_norm)

# Make an hmm and sample from it
arhmm = ssm.HMM(K, D, observations="ar")
arhmm.fit(pca_proj[:N_FRAMES, :])
zhat = arhmm.most_likely_states(pca_proj[:N_FRAMES, :])

# Get transition matrix
transition_mat = arhmm.transitions.transition_matrix

# %% Plot
color_names = ["faded green", "red", "amber", "windows blue"]
cmap = ListedColormap(sns.xkcd_palette(color_names))
colors, dpi = figure_style()

time_ax = np.linspace(0, N_FRAMES/FRAME_RATE, N_FRAMES)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.5, 1.75),
                                    gridspec_kw={'width_ratios':[1, 1, 0.7]}, dpi=dpi)
ax1.imshow(zhat[None,:], aspect="auto",
           extent=[0, N_FRAMES/FRAME_RATE, pca_proj[:N_FRAMES, 0].min(), pca_proj[:N_FRAMES, 0].max()],
           cmap=cmap, vmin=0, vmax=K)
ax1.plot(time_ax, pca_proj[:N_FRAMES, 0], zorder=1, color='k', lw=0.5)
ax1.set(xlabel='Time (s)', ylabel='PC 1')

ax2.imshow(zhat[None,:], aspect="auto",
           extent=[0, N_FRAMES/FRAME_RATE, pca_proj[:N_FRAMES, 0].min(), pca_proj[:N_FRAMES, 0].max()],
           cmap=cmap, vmin=0, vmax=K)
ax2.plot(time_ax, pca_proj[:N_FRAMES, 0], zorder=1, color='k', lw=0.5)
ax2.set(xlabel='Time (s)', ylabel='PC 1', xlim=[0, 5])

im = ax3.imshow(transition_mat, cmap='gray')
ax3.set(title="Transition Matrix")

cbar_ax = fig.add_axes([0.955, 0.25, 0.01, 0.6])
fig.colorbar(im, cax=cbar_ax)

plt.tight_layout()
