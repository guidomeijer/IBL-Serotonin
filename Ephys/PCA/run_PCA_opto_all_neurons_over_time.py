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
from brainbox.metrics.single_units import spike_sorting_metrics
from serotonin_functions import figure_style
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from brainbox.io.one import SpikeSortingLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from serotonin_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                                 get_artifact_neurons, get_neuron_qc)
from one.api import ONE
from ibllib.atlas import AllenAtlas
one = ONE()
ba = AllenAtlas()
pca = PCA(n_components=10)

# Settings
NEURON_QC = True
T_BEFORE = 1
T_AFTER = 4
BASELINE = [-1, 0]
BIN_SIZE = 0.02
SMOOTHING = 0.02
MIN_FR = 0.1
PLOT = True
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'PCA', 'AllNeurons')

# Get binning time vectors
BIN_CENTERS = np.arange(-T_BEFORE, T_AFTER, BIN_SIZE) + (BIN_SIZE / 2)

# Query sessions
rec = query_ephys_sessions(one=one)

pca_df = pd.DataFrame()
pca_dist_df = pd.DataFrame()
artifact_neurons = get_artifact_neurons()
for i, eid in enumerate(np.unique(rec['eid'])):

    # Get session details
    subject = np.unique(rec.loc[rec['eid'] == eid, 'subject'])[0]
    date = np.unique(rec.loc[rec['eid'] == eid, 'date'])[0]
    print(f'Starting {subject}, {date}')

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
            qc_metrics = get_neuron_qc(pid, one=one, ba=ba)
            clusters_pass[probe] = np.where(qc_metrics['label'] == 1)[0]
        else:
            clusters_pass[probe] = np.unique(spikes.clusters)
        clusters_pass[probe] = clusters_pass[probe][~np.isin(clusters_pass[probe], artifact_neurons.loc[
            artifact_neurons['pid'] == pid, 'neuron_id'].values)]
        clusters[probe]['region'] = remap(clusters[probe]['acronym'], combine=True, abbreviate=True)

        # Create population activity arrays for opto stim activity
        pca_opto, spks_opto = dict(), dict()
        for probe in spikes.keys():
            # Get smoothed firing rates
            peths, _ = calculate_peths(spikes[probe].times, spikes[probe].clusters, clusters_pass[probe],
                                       opto_train_times, T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING)
            if probe == 'probe00':
                pop_opto = peths['means'].T
                time = peths['tscale']
            elif probe == 'probe01':
                pop_opto = np.concatenate((pop_opto, peths['means'].T), axis=1)

        # Create population activity arrays for random onset times in the spontaneous activity
        spont_times = np.random.uniform(low=opto_train_times[0] - (6*60), high=opto_train_times-1,
                                        size=(opto_train_times.shape[0]))
        pca_spont, spks_spont = dict(), dict()
        for probe in spikes.keys():
            # Get smoothed firing rates
            peths, _ = calculate_peths(spikes[probe].times, spikes[probe].clusters, clusters_pass[probe],
                                       spont_times, T_BEFORE, T_AFTER, BIN_SIZE, SMOOTHING)
            if probe == 'probe00':
                pop_spont = peths['means'].T
                time = peths['tscale']
            elif probe == 'probe01':
                pop_spont = np.concatenate((pop_spont, peths['means'].T), axis=1)

    # Opto stim activity
    # Normalize data
    ss = StandardScaler(with_mean=True, with_std=True)
    pop_opto_norm = ss.fit_transform(pop_opto)

    # Perform PCA
    pca_opto = pca.fit_transform(pop_opto_norm)
    pca_opto_baseline = np.mean(pca_opto[(time >= BASELINE[0]) & (time < BASELINE[1]), :], axis=0)
    pca_dist_opto = np.empty(pca_opto.shape[0])
    for kk in range(pca_opto.shape[0]):
        pca_dist_opto[kk] = np.linalg.norm(pca_opto_baseline - pca_opto[kk, :])


    # Spontaneous activity
    # Normalize data
    ss = StandardScaler(with_mean=True, with_std=True)
    pop_spont_norm = ss.fit_transform(pop_spont)

    # Perform PCA
    pca_spont = pca.fit_transform(pop_spont_norm)
    pca_spont_baseline = np.mean(pca_spont[(time >= BASELINE[0]) & (time < BASELINE[1]), :], axis=0)
    pca_dist_spont = np.empty(pca_spont.shape[0])
    for kk in range(pca_spont.shape[0]):
        pca_dist_spont[kk] = np.linalg.norm(pca_spont_baseline - pca_spont[kk, :])

    # Distance from baseline in normal neural space
    pop_opto_baseline = np.mean(pop_opto[(time >= BASELINE[0]) & (time < BASELINE[1]), :], axis=0)
    pop_dist_opto = np.empty(pop_opto.shape[0])
    for kk in range(pop_opto.shape[0]):
        pop_dist_opto[kk] = np.linalg.norm(pop_opto_baseline - pop_opto[kk, :])

    # Plot result
    if PLOT:
        colors, dpi = figure_style()
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(8, 2.5), dpi=dpi)
        ax1.plot(time, pca_opto[:, 0])
        ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='grey')
        ax1.set(ylabel='First principal component', xlabel='Time (s)')

        ax2.scatter(pca_opto[:, 0], pca_opto[:, 1], c=time, cmap='twilight_r')
        ax2.set(xlabel='PC 1', ylabel='PC 2')

        ax3.plot(time, np.sum(pca_opto, axis=1))
        ax3.plot([0, 0], ax3.get_ylim(), ls='--', color='grey')
        ax3.set(xlabel='Time (s)', ylabel='Summed first 3 PCs')

        ax4.plot(time, pca_dist_opto)
        ax4.plot([0, 0], ax4.get_ylim(), ls='--', color='grey')
        ax4.set(xlabel='Time (s)', ylabel='PCA distance')

        sns.despine(trim=True)
        plt.tight_layout()
        plt.savefig(join(fig_path, f'{subject}_{date}.pdf'), dpi=300)
        plt.close(f)

    # Add to dataframe
    pca_df = pd.concat((pca_df, pd.DataFrame(data={
        'subject': subject, 'date': date, 'eid': eid, 'time': time, 'n_neurons': pop_opto.shape[1],
        'pca_dist': pca_dist_opto, 'pca_dist_spont': pca_dist_spont, 'pop_dist_opto': pop_dist_opto,
        'pca1': pca_opto[:, 0], 'pca2': pca_opto[:, 1], 'pca3': pca_opto[:, 2]})), ignore_index=True)

    pca_df.to_csv(join(save_path, 'pca_all_neurons.csv'), index=False)

pca_df.to_csv(join(save_path, 'pca_all_neurons.csv'), index=False)

