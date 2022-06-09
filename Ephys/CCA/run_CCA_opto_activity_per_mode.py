#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
from scipy.stats import pearsonr
from scipy.signal import gaussian
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from brainbox.metrics.single_units import spike_sorting_metrics
from brainbox.io.one import SpikeSortingLoader
from serotonin_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                                 get_artifact_neurons, calculate_peths)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
REGION_PAIRS = [['M2', 'mPFC'], ['M2', 'ORB']]
FIT_TIME = 1.2  # timewindow relative to opto stim to use to fit CCA axes
N_MODES = 5  # number of CCA modes to calculate
NEURON_QC = True  # whether to use neuron qc to exclude bad units
MIN_NEURONS = 10  # minimum neurons per region
WIN_SIZE = 0.05  # window size in seconds
PRE_TIME = 1  # time before stim onset in s
POST_TIME = 3  # time after stim onset in s
SMOOTHING = 0.1  # smoothing of psth
SUBTRACT_MEAN = True  # whether to subtract the mean PSTH from each trial
MIN_FR = 0.5  # minimum firing rate over the whole recording
N_PC = 10  # number of PCs to use

# Paths
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'CCA', 'RegionPairs')

# Initialize
cca = CCA(n_components=N_MODES, max_iter=1000)
pca = PCA(n_components=10)
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

cca_df = pd.DataFrame()
for i, eid in enumerate(np.unique(rec['eid'])):

    # Get session details
    subject = rec.loc[rec['eid'] == eid, 'subject'].values[0]
    date = rec.loc[rec['eid'] == eid, 'date'].values[0]
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
            sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
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
    pca_opto, pca_fit = dict(), dict()
    for probe in spikes.keys():
        for region in np.unique(REGION_PAIRS):

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
                     post_time=POST_TIME, bin_size=WIN_SIZE, smoothing=SMOOTHING, return_fr=False)

                 if SUBTRACT_MEAN:
                     # Subtract mean PSTH from each opto stim
                     for tt in range(binned_spks_opto.shape[0]):
                         binned_spks_opto[tt, :, :] = binned_spks_opto[tt, :, :] - psth_opto['means']

                 # Perform PCA
                 pca_opto[region] = np.empty([binned_spks_opto.shape[0], N_PC, binned_spks_opto.shape[2]])
                 pca_fit[region] = np.empty(binned_spks_opto.shape[2]).astype(object)
                 for tb in range(binned_spks_opto.shape[2]):
                     pca_fit[region][tb] = pca.fit(binned_spks_opto[:, :, tb])
                     pca_opto[region][:, :, tb] = pca_fit[region][tb].transform(binned_spks_opto[:, :, tb])

    # Perform CCA per region pair
    print('Starting CCA per region pair')
    all_cca_df = pd.DataFrame()
    for r1, region_1 in enumerate(pca_opto.keys()):
        for r2, region_2 in enumerate(list(pca_opto.keys())[r1:]):
            if region_1 == region_2:
                continue

            # Run CCA per region pair
            print(f'{region_1}-{region_2}')
            act_modes = np.empty((N_MODES, n_time_bins))

            # Fit CCA axis to the specified timebin
            fit_tb = np.argmin(np.abs(FIT_TIME - psth_opto['tscale']))
            cca.fit(pca_opto[region_1][:, :, fit_tb], pca_opto[region_2][:, :, fit_tb])

            for tb in range(n_time_bins):

                # Project data to fitted CCA axis
                x, y = cca.transform(pca_opto[region_1][:, :, tb], pca_opto[region_2][:, :, tb])

                # Inverse projected CCA scores back to PCA space and then back to neural activity space
                for mm in range(N_MODES):
                    inverse_pca_proj = np.outer(x[:, mm], cca.x_weights_[:, mm])
                    neural_act = pca_fit[region_1][tb].inverse_transform(inverse_pca_proj)

                    # Normalize activity of each neuron and get mean
                    norm_act = neural_act.copy()
                    for nn in range(neural_act.shape[1]):
                        norm_act[:, nn] = neural_act[:, nn] + np.abs(neural_act[:, nn].min())
                        norm_act[:, nn] = norm_act[:, nn] / norm_act[:, nn].max()
                    act_modes[mm, tb] = np.nanmean(norm_act)

            # Baseline subtract
            act_baseline = act_modes.copy()
            for mm in range(N_MODES):
                act_baseline[mm, :] = act_modes[mm, :] - np.mean(act_modes[mm, psth_opto['tscale'] < 0])

            # Add to dataframe
            cca_df = pd.concat((cca_df, pd.DataFrame(index=[cca_df.shape[0]+1], data={
                'subject': subject, 'date': date, 'eid': eid, 'region_1': region_1, 'region_2': region_2,
                'region_pair': f'{region_1}-{region_2}', 'r_opto': [act_modes], 'r_baseline': [act_baseline],
                'time': [psth_opto['tscale']]})), ignore_index=True)

        # Save results
        cca_df.to_pickle(join(save_path, 'cca_activity_per_mode_opto.pkl'))

