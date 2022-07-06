# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 19:34:20 2022

@author: Guido
"""

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
from brainbox.io.one import SpikeSortingLoader
from serotonin_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                                 get_artifact_neurons, get_neuron_qc, figure_style,
                                 calculate_peths)
from one.api import ONE
from ibllib.atlas import AllenAtlas
one = ONE()
ba = AllenAtlas()

# Settings
T_BEFORE = 1
T_AFTER = 3
BIN_SIZE = 0.25
SMOOTHING = 0
SUBTRACT_MEAN = False
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'PCA')


# Query sessions
rec = query_ephys_sessions(one=one, acronym=['MOs'])

artifact_neurons = get_artifact_neurons()
corr_df = pd.DataFrame()
for i in rec.index.values:

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
    try:
        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
    except:
        continue

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
    cluster_ids = clusters['cluster_id'][clusters_pass]

    # Get peri-event time histogram
    peths, spike_counts = calculate_peths(spikes.times, spikes.clusters, cluster_ids,
                                          opto_train_times, T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING,
                                          return_fr=False)
    tscale = peths['tscale']


    if SUBTRACT_MEAN:
        # Subtract mean PSTH from each opto stim
        for tt in range(spike_counts.shape[0]):
            spike_counts[tt, :, :] = spike_counts[tt, :, :] - peths['means']

    # Get noise correlations
    if np.sum(clusters_regions == 'M2') == 0:
        continue
    if np.sum(clusters_regions == 'OFC') > 0:
        all_r, region_1, region_2 = [], [], []
        for ind1, id1 in enumerate(np.where(clusters_regions == 'M2')[0]):
            for ind2, id2 in enumerate(np.where(clusters_regions == 'OFC')[0]):
                r = np.empty(spike_counts.shape[2])
                for indtb in range(spike_counts.shape[2]):
                    r[indtb] = pearsonr(spike_counts[:, ind1, indtb], spike_counts[:, ind2, indtb])[0]
                r = r - np.mean(r[tscale < 0])
                all_r.append(r)
        region_corr = np.vstack(all_r)

        # Add to df
        corr_df = pd.concat((corr_df, pd.DataFrame(data={
            'pid': pid, 'subject': subject, 'date': date, 'region_pair': 'M2-OFC',
            'region_corr': np.nanmean(region_corr, axis=0), 'time': tscale})), ignore_index=True)

    # Get noise correlations
    if np.sum(clusters_regions == 'mPFC') > 0:
        all_r, region_1, region_2 = [], [], []
        for ind1, id1 in enumerate(np.where(clusters_regions == 'M2')[0]):
            for ind2, id2 in enumerate(np.where(clusters_regions == 'mPFC')[0]):
                r = np.empty(spike_counts.shape[2])
                for indtb in range(spike_counts.shape[2]):
                    r[indtb] = pearsonr(spike_counts[:, ind1, indtb], spike_counts[:, ind2, indtb])[0]
                r = r - np.mean(r[tscale < 0])
                all_r.append(r)
        region_corr = np.vstack(all_r)

    # Add to df
    corr_df = pd.concat((corr_df, pd.DataFrame(data={
        'pid': pid, 'subject': subject, 'date': date, 'region_pair': 'M2-mPFC',
        'region_corr': np.nanmean(region_corr, axis=0), 'time': tscale})), ignore_index=True)

corr_df.to_csv(join(save_path, 'region_corr_frontal.csv'))