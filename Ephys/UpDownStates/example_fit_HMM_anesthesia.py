#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:00:46 2022
By: Guido Meijer
"""

import numpy as np
import seaborn as sns
import ssm
import matplotlib.pyplot as plt
from os.path import join
from brainbox.processing import bincount2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from serotonin_functions import (figure_style, get_neuron_qc, load_passive_opto_times, paths,
                                 high_level_regions)
from brainbox.io.one import SpikeSortingLoader
from matplotlib.colors import ListedColormap
from brainbox.singlecell import calculate_peths
from ibllib.atlas import AllenAtlas
from one.api import ONE
one = ONE()
ba = AllenAtlas()

K = 2    # number of discrete states
do_PCA = True
D = 10   # dimensions of PCA
BIN_SIZE = 0.25
SMOOTHING = 0
PID = '04954136-75a8-4a20-9054-37b0bffd3b8b'
fig_path, _ = paths()

# Load in opto times
opto_times, _ = load_passive_opto_times(one.pid2eid(PID)[0], anesthesia=True, one=one)

# Load in neural data
sl = SpikeSortingLoader(pid=PID, one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)

# Remap to high level regions
clusters.regions = high_level_regions(clusters.acronym, merge_cortex=True)

# Filter neurons that pass QC and are in cortex, thalamus or striatum
qc_metrics = get_neuron_qc(PID, one=one, ba=ba)
#clusters_pass = np.where((qc_metrics['label'] == 1)
#                         & (np.in1d(clusters.regions, ['Cortex', 'Striatum', 'Thalamus'])))[0]
clusters_pass = np.where(qc_metrics['label'] == 1)[0]
spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
spikes.depths = spikes.depths[np.isin(spikes.clusters, clusters_pass)]
spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]

# Get smoothed firing rates
peth, _ = calculate_peths(spikes.times, spikes.clusters, np.unique(spikes.clusters),
                          [opto_times[0]-300], pre_time=0, post_time=300,
                          bin_size=BIN_SIZE, smoothing=SMOOTHING)
tscale = peth['tscale'] - peth['tscale'][0]
time_ax = peth['tscale'] + opto_times[0]-300
pop_act = peth['means'].T

# Do PCA
if do_PCA:
    pca = PCA(n_components=D)
    ss = StandardScaler(with_mean=True, with_std=True)
    pop_vector_norm = ss.fit_transform(pop_act)
    pca_proj = pca.fit_transform(pop_vector_norm)

    # Make an hmm and sample from it
    arhmm = ssm.HMM(K, pca_proj.shape[1], observations="gaussian")
    arhmm.fit(pca_proj)
    zhat = arhmm.most_likely_states(pca_proj)

else:
    # Make an hmm and sample from it
    arhmm = ssm.HMM(K, pop_act.shape[1], observations="gaussian")
    arhmm.fit(pop_act)
    zhat = arhmm.most_likely_states(pop_act)

# Get transition matrix
transition_mat = arhmm.transitions.transition_matrix

# Get spike raster
R, times, depths = bincount2D(spikes.times, spikes.depths, xbin=0.01, ybin=20, weights=None)

# %% Plot
color_names = ["faded green", "amber"]
cmap = ListedColormap(sns.xkcd_palette(color_names))
colors, dpi = figure_style()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.5, 1.75),
                                    gridspec_kw={'width_ratios':[1, 1, 0.7]}, dpi=dpi)

ax1.imshow(zhat[None,tscale <= 120], aspect="auto", extent=[0, 120, 0, 7], cmap=cmap, vmin=0, vmax=K-1)
ax1.plot(tscale[tscale <= 120], np.mean(pop_act[tscale <= 120, :], axis=1), zorder=1, color='k', lw=0.5)
ax1.set(xlabel='Time (s)', ylabel='Population activity (spks/s)')

ax2.imshow(zhat[None,tscale <= 10], aspect="auto", extent=[0, 10, 0, 7], cmap=cmap, vmin=0, vmax=K-1)
ax2.plot(tscale[tscale <= 10], np.mean(pop_act[tscale <= 10, :], axis=1), zorder=1, color='k', lw=0.5)
ax2.set(xlabel='Time (s)', ylabel='Population activity (spks/s)')

im = ax3.imshow(transition_mat, cmap='gray')
ax3.set(title="Transition Matrix")

cbar_ax = fig.add_axes([0.955, 0.25, 0.01, 0.6])
fig.colorbar(im, cax=cbar_ax)

plt.tight_layout()
plt.savefig(join(fig_path, 'Ephys', 'Anesthesia', 'up_down_classification.jpg'), dpi=600)

# %%

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), dpi=dpi)

ax1.imshow(R, aspect='auto', cmap='binary', vmin=0, vmax=np.std(R),
           extent=np.r_[times[[0, -1]], depths[[0, -1]]], origin='upper')
ax1.set(xlim=[time_ax[0], time_ax[0] + 100], ylim=[0, 4000],
        xlabel='Time since start of recording (s)', ylabel='Depth (um)')
ax1.invert_yaxis()

ax2.imshow(R, aspect='auto', cmap='binary', vmin=0, vmax=np.std(R),
           extent=np.r_[times[[0, -1]], depths[[0, -1]]], origin='upper')
ax2.imshow(zhat[None,:], aspect="auto", extent=[time_ax[0], time_ax[-1], 0, 4000],
           cmap=cmap, vmin=0, vmax=K-1, alpha=0.25)
ax2.set(xlim=[time_ax[0], time_ax[0] + 100], ylim=[0, 4000],
        xlabel='Time since start of recording (s)', ylabel='Depth (um)')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig(join(fig_path, 'Ephys', 'Anesthesia', 'up_down_raster.jpg'), dpi=600)