#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from brainbox.singlecell import calculate_peths
from serotonin_functions import figure_style
from brainbox.io.one import SpikeSortingLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from serotonin_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                                 get_artifact_neurons, get_neuron_qc)
from one.api import ONE
from ibllib.atlas import AllenAtlas
one = ONE()
ba = AllenAtlas()
pca = PCA(n_components=3)

# Settings
MIN_NEURONS = 10  # per region
T_BEFORE = 1
T_AFTER = 3
BASELINE = [-0.5, 0]
BIN_SIZE = 0.02
SMOOTHING = 0.02
MIN_FR = 0.1
PLOT = False
OVERWRITE = True
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'PCA')

# Get binning time vectors
BIN_CENTERS = np.arange(-T_BEFORE, T_AFTER, BIN_SIZE) + (BIN_SIZE / 2)

# Query sessions
rec = query_ephys_sessions(one=one)

if OVERWRITE:
    pca_df = pd.DataFrame()
    pca_dist_df = pd.DataFrame()
    rec_ind = rec.index.values
else:
    pca_df = pd.read_csv(join(save_path, 'pca_regions.csv'))
    pca_dist_df = pd.read_csv(join(save_path, 'pca_dist_regions.csv'))
    rec_ind = [i for i in rec.index.values if rec.loc[i, 'pid'] not in pca_df['pid'].values]
artifact_neurons = get_artifact_neurons()
for i in rec_ind:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']
    print(f'Starting {subject}, {date}, {probe}')

    # Load in laser pulse times
    try:
        opto_train_times, _ = load_passive_opto_times(eid, one=one)
    except:
        print('Session does not have passive laser pulses')
        continue
    if len(opto_train_times) == 0:
        print('Did not find ANY laser pulses!')
        continue
    else:
        print(f'Found {len(opto_train_times)} passive laser pulses')

    # Load in spikes
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    # Filter neurons that pass QC
    qc_metrics = get_neuron_qc(pid, one=one, ba=ba)
    clusters_pass = np.where(qc_metrics['label'] == 1)[0]

    # Exclude artifact neurons
    clusters_pass = np.array([i for i in clusters_pass if i not in artifact_neurons.loc[
        artifact_neurons['pid'] == pid, 'neuron_id'].values])
    if clusters_pass.shape[0] == 0:
            continue

    # Select QC pass neurons
    spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
    spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]
    clusters_pass = clusters_pass[np.isin(clusters_pass, np.unique(spikes.clusters))]

    # Get regions from Beryl atlas
    clusters['region'] = remap(clusters['acronym'], combine=True)
    clusters_regions = clusters['region'][clusters_pass]

    # Loop over regions
    for r, region in enumerate(np.unique(clusters_regions)):
        if region == 'root':
            continue

        # Select spikes and clusters in this brain region
        clusters_in_region = clusters_pass[clusters_regions == region]
        spks_region = spikes.times[np.isin(spikes.clusters, clusters_in_region)]
        clus_region = spikes.clusters[np.isin(spikes.clusters, clusters_in_region)]
        if len(clusters_in_region) < MIN_NEURONS:
            continue

        # Exclude neurons with too low firing rates
        excl_neurons = []
        for n, neuron_id in enumerate(clusters_in_region):
            if np.sum(clus_region == neuron_id) / 360 < MIN_FR:
                excl_neurons.append(neuron_id)
        clusters_in_region = np.setdiff1d(clusters_in_region, excl_neurons)

        # Get smoothed firing rates
        peths, _ = calculate_peths(spks_region, clus_region, clusters_in_region,
                                   opto_train_times, T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING)
        pop_vector = peths['means'].T
        time = peths['tscale']

        # Normalize data
        ss = StandardScaler(with_mean=True, with_std=True)
        pop_vector_norm = ss.fit_transform(pop_vector)

        # Perform PCA
        pca_proj = pca.fit_transform(pop_vector_norm)

        # Get Eucledian distance between PCA points
        pca_diff = np.diff(pca_proj[:, :2], axis=0)
        pca_dist = np.empty(pca_diff.shape[0])
        for kk in range(pca_diff.shape[0]):
            pca_dist[kk] = np.linalg.norm(pca_diff[kk, :])
        time_diff = (time[:-1] + time[1:]) / 2

        # Plot result
        colors, dpi = figure_style()
        """
        f = plt.figure(figsize=(6, 3), dpi=dpi)
        ax = plt.axes(projection='3d')
        p = ax.scatter3D(pca_proj[:, 0], pca_proj[:, 1], pca_proj[:, 2], c=peths['tscale'],
                         cmap='twilight_r')
        ax.set(xlabel='PC 1', ylabel='PC 2', zlabel='PC 3', title=f'{region}')
        axins = inset_axes(ax, width="5%", height="80%", loc='upper left',
                           bbox_to_anchor=(1.3, 0., 1, 1),
                           bbox_transform=ax.transAxes,
                           borderpad=0,)
        cbar = f.colorbar(p, cax=axins, shrink=0.6)
        cbar.ax.set_ylabel('Time (s)', rotation=270, labelpad=18)
        plt.savefig(join(fig_path, 'SinglePlots', f'{region}_{subject}_{date}_3D.jpg'), dpi=300)
        plt.savefig(join(fig_path, 'SinglePlots', f'{region}_{subject}_{date}_3D.pdf'))
        plt.close(f)
        """
        if PLOT:
            f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(8, 2.5), dpi=dpi)
            ax1.plot(time, pca_proj[:, 0])
            ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='grey')
            ax1.set(ylabel='First principal component', xlabel='Time (s)', title=f'{region}')

            ax2.scatter(pca_proj[:, 0], pca_proj[:, 1], c=time, cmap='twilight_r')
            ax2.set(xlabel='PC 1', ylabel='PC 2', title=f'{region}')

            ax3.plot(time, np.sum(pca_proj, axis=1))
            ax3.plot([0, 0], ax3.get_ylim(), ls='--', color='grey')
            ax3.set(xlabel='Time (s)', ylabel='Summed first 3 PCs')

            ax4.plot(time_diff, pca_dist)
            ax4.plot([0, 0], ax4.get_ylim(), ls='--', color='grey')
            ax4.set(xlabel='Time (s)', ylabel='PCA distance')

            sns.despine(trim=True)
            plt.tight_layout()
            plt.savefig(join(fig_path, 'SinglePlots', f'{region}_{subject}_{date}.jpg'), dpi=300)
            plt.close(f)

        # Add to dataframe
        pca_df = pd.concat((pca_df, pd.DataFrame(data={
            'subject': subject, 'date': date, 'pid': pid, 'probe': probe, 'region': region,
            'time': time, 'n_neurons': np.sum(clusters_regions == region),
            'pca1': pca_proj[:, 0], 'pca2': pca_proj[:, 1], 'pca3': pca_proj[:, 2]})))
        pca_dist_df = pd.concat((pca_dist_df, pd.DataFrame(data={
            'subject': subject, 'date': date, 'pid': pid, 'probe': probe, 'region': region,
            'time': time_diff, 'n_neurons': np.sum(clusters_regions == region),
            'pca_dist': pca_dist})))

    pca_df.to_csv(join(save_path, 'pca_regions.csv'), index=False)
    pca_dist_df.to_csv(join(save_path, 'pca_dist_regions.csv'), index=False)
pca_df = pca_df.reset_index()
pca_df.to_csv(join(save_path, 'pca_regions.csv'), index=False)
pca_dist_df.to_csv(join(save_path, 'pca_dist_regions.csv'), index=False)

