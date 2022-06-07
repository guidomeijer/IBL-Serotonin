#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join, isfile
import pandas as pd
from scipy.stats import pearsonr
from scipy.signal import gaussian
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, KFold
from brainbox.metrics.single_units import spike_sorting_metrics
from brainbox.io.one import SpikeSortingLoader
from serotonin_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                                 get_artifact_neurons, calculate_peths, figure_style)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()
cca = CCA(n_components=1, max_iter=1000)
pca = PCA(n_components=10)

# Settings
eid = '0d04401a-2e75-4449-b699-252000ed2b76'
region_1 = 'M2'
region_2 = 'ORB'
time_1 = 0.755
time_2 = 0.855
NEURON_QC = True  # whether to use neuron qc to exclude bad units
MIN_NEURONS = 10  # minimum neurons per region
WIN_SIZE = 0.01  # window size in seconds
PRE_TIME = 1.25  # time before stim onset in s
POST_TIME = 3.25  # time after stim onset in s
SMOOTHING = 0.02  # smoothing of psth
MIN_FR = 0.5  # minimum firing rate over the whole recording
N_PC = 10  # number of PCs to use

# Paths
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'CCA')

# Initialize some things
np.random.seed(42)  # fix random seed for reproducibility
n_time_bins = int((PRE_TIME + POST_TIME) / WIN_SIZE)
if SMOOTHING > 0:
    w = n_time_bins - 1 if n_time_bins % 2 == 0 else n_time_bins
    window = gaussian(w, std=SMOOTHING / WIN_SIZE)
    window /= np.sum(window)

# Query sessions with frontal and amygdala
rec = query_ephys_sessions(one=one)

# Load in artifact neurons
artifact_neurons = get_artifact_neurons()

# Get session details
subject = rec.loc[rec['eid'] == eid, 'subject'].values[0]
date = rec.loc[rec['eid'] == eid, 'date'].values[0]
print(f'Starting {subject}, {date}')

# Load in laser pulse times
opto_train_times, _ = load_passive_opto_times(eid, one=one)

# Load in neural data of both probes of the recording
spikes, clusters, channels, clusters_pass = dict(), dict(), dict(), dict()
for (pid, probe) in zip(rec.loc[rec['eid'] == eid, 'pid'].values, rec.loc[rec['eid'] == eid, 'probe'].values):

    try:
        sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
        spikes[probe], clusters[probe], channels[probe] = sl.load_spike_sorting()
        clusters[probe] = sl.merge_clusters(spikes[probe], clusters[probe], channels[probe])
    except Exception as err:
        print(err)
        continue

    # Filter neurons that pass QC and artifact neurons
    if NEURON_QC:
        print('Calculating neuron QC metrics..')
        qc_metrics, _ = spike_sorting_metrics(spikes[probe].times, spikes[probe].clusters,
                                              spikes[probe].amps, spikes[probe].depths,
                                              cluster_ids=np.arange(clusters[probe].channels.size))
        clusters_pass[probe] = np.where(qc_metrics['label'] > 0.5)[0]
    else:
        clusters_pass[probe] = np.unique(spikes.clusters)
    clusters_pass[probe] = clusters_pass[probe][~np.isin(clusters_pass[probe], artifact_neurons.loc[
        artifact_neurons['pid'] == pid, 'neuron_id'].values)]
    clusters[probe]['region'] = remap(clusters[probe]['acronym'], combine=True, abbreviate=True)

# Create population activity arrays for all regions
pca_opto, spks_opto, spks_residuals = dict(), dict(), dict()
for probe in spikes.keys():
    for region in [region_1, region_2]:

        # Exclude neurons with low firing rates
        clusters_in_region = np.where(clusters[probe]['region'] == region)[0]
        fr = np.empty(clusters_in_region.shape[0])
        for nn, neuron_id in enumerate(clusters_in_region):
            fr[nn] = np.sum(spikes[probe].clusters == neuron_id) / spikes[probe].clusters[-1]
        clusters_in_region = clusters_in_region[fr >= MIN_FR]

        # Get spikes and clusters
        spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)
                                          & np.isin(spikes[probe].clusters, clusters_pass[probe])]
        clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_in_region)
                                             & np.isin(spikes[probe].clusters, clusters_pass[probe])]

        if (len(np.unique(clus_region)) >= MIN_NEURONS) & (region != 'root'):
            print(f'Loading population activity for {region}')

            # Get PSTH and binned spikes for OPTO activity
            psth_opto, binned_spks_opto = calculate_peths(
                spks_region, clus_region, np.unique(clus_region), opto_train_times, pre_time=PRE_TIME,
                post_time=POST_TIME, bin_size=WIN_SIZE, smoothing=SMOOTHING, return_fr=True)

           
            binned_spks_residuals = binned_spks_opto.copy() 
            # Subtract mean PSTH from each opto stim
            for tt in range(binned_spks_opto.shape[0]):
                binned_spks_residuals[tt, :, :] = binned_spks_opto[tt, :, :] - psth_opto['means']

            # Add to dict
            spks_opto[region] = binned_spks_opto
            spks_residuals[region] = binned_spks_residuals

            # Perform PCA
            pca_opto[region] = np.empty([binned_spks_residuals.shape[0], N_PC, binned_spks_residuals.shape[2]])
            for tb in range(binned_spks_residuals.shape[2]):
                pca_opto[region][:, :, tb] = pca.fit_transform(binned_spks_residuals[:, :, tb])

# Get timebin index
tb_1 = np.where(psth_opto['tscale'] == time_1)[0][0]
tb_2 = np.where(psth_opto['tscale'] == time_2)[0][0]

# Create indices for odd and even trials
even_ind = np.arange(0, pca_opto[region_1][:, :, 0].shape[0], 2).astype(int)
odd_ind = np.arange(1, pca_opto[region_1][:, :, 0].shape[0], 2).astype(int)

# Fit on the even trials and correlate the odd trials
cca.fit(pca_opto[region_1][even_ind, :, tb_1],
        pca_opto[region_2][even_ind, :, tb_2])
x, y = cca.transform(pca_opto[region_1][odd_ind, :, tb_1],
                     pca_opto[region_2][odd_ind, :, tb_2])
r_splits = []
r_splits.append(pearsonr(x.T[0], y.T[0])[0])

# Fit on the odd trials and correlate the even trials
cca.fit(pca_opto[region_1][odd_ind, :, tb_1],
        pca_opto[region_2][odd_ind, :, tb_2])
x, y = cca.transform(pca_opto[region_1][even_ind, :, tb_1],
                     pca_opto[region_2][even_ind, :, tb_2])
r_splits.append(pearsonr(x.T[0], y.T[0])[0])
print(f'r={np.mean(r_splits)}')

# %% Plot
colors, dpi = figure_style()
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(5, 3.8), dpi=dpi)

plt_hndl = ax1.imshow(spks_residuals['M2'][:,:,tb_1], cmap='viridis')
cbar = plt.colorbar(plt_hndl, ax=ax1, shrink=0.5)
cbar.set_label('spks/s', rotation=270)
cbar.ax.get_yaxis().labelpad = 8
ax1.set(title=f'Secondary motor cortex \n Mean subtracted firing rates \n t = {time_1} s after stim onset',
        ylabel='Trials', yticks=[1, spks_residuals['M2'].shape[0]], xticks=[1, spks_residuals['M2'].shape[1]])
ax1.set_xlabel('Neurons', labelpad=-5)

plt_hndl = ax2.imshow(pca_opto['M2'][:,:,tb_1], cmap='coolwarm', vmin=-50, vmax=50)
plt.colorbar(plt_hndl, ax=ax2, shrink=0.5)
ax2.set(ylabel='Trials', xlabel='PC dims.', xticks=[1, 10],
        yticks=[1, spks_residuals['M2'].shape[0]])

plt_hndl = ax3.imshow(spks_residuals['ORB'][:,:,tb_1], cmap='viridis')
cbar = plt.colorbar(plt_hndl, ax=ax3, shrink=0.5)
cbar.set_label('spks/s', rotation=270)
cbar.ax.get_yaxis().labelpad = 8
ax3.set(title=f'Orbitofrontal cortex \n Mean subtracted firing rates \n t = {time_2} s after stim onset',
        ylabel='Trials', yticks=[1, spks_residuals['ORB'].shape[0]], xticks=[1, spks_residuals['ORB'].shape[1]])
ax3.set_xlabel('Neurons', labelpad=-5)

plt_hndl = ax4.imshow(pca_opto['ORB'][:,:,tb_1], cmap='coolwarm', vmin=-50, vmax=50)
plt.colorbar(plt_hndl, ax=ax4, shrink=0.5)
ax4.set(ylabel='Trials', xlabel='PC dims.', xticks=[1, 10],
        yticks=[1, spks_residuals['ORB'].shape[0]])

plt.tight_layout()
sns.despine(trim=True)

plt.savefig(join(fig_path, 'CCA_Example.pdf'))

