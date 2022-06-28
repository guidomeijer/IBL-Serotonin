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
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, KFold
from brainbox.metrics.single_units import spike_sorting_metrics
from brainbox.io.one import SpikeSortingLoader
from serotonin_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                                 get_artifact_neurons, calculate_peths)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()
pca = PCA(n_components=10)

# Settings
OVERWRITE = True  # whether to overwrite existing runs
NEURON_QC = False  # whether to use neuron qc to exclude bad units
PCA = True  # whether to use PCA on neural activity before CCA
N_PC = 10  # number of PCs to use
MIN_NEURONS = 10  # minimum neurons per region
WIN_SIZE = 0.05  # window size in seconds
PRE_TIME = 1.5  # time before stim onset in s
POST_TIME = 3.5  # time after stim onset in s
SMOOTHING = 0.1  # smoothing of psth
FIT_WIN = [0, 2]  # window to fit axis (rel to opto onset)
MIN_FR = 0.5  # minimum firing rate over the whole recording
N_MODES = 10

# Paths
fig_path, save_path = paths()

# Initialize some things
REGION_PAIRS = [['M2', 'mPFC'], ['M2', 'ORB'], ['mPFC', 'Amyg'], ['ORB', 'Amyg'], ['M2', 'Amyg'],
                ['Hipp', 'PPC'], ['Hipp', 'Thal'], ['ORB', 'mPFC'], ['PPC', 'Thal'], ['MRN', 'SC'],
                ['RSP', 'SC'], ['BC', 'Str'], ['MRN', 'RSP'], ['MRN', 'SN'], ['Pir', 'Str'],
                ['SC', 'SN']]
cca = CCA(n_components=N_MODES, max_iter=1000)
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

file_name = 'CCA_opto_window.pickle'
if ~OVERWRITE & isfile(join(save_path, file_name)):
    cca_df = pd.read_pickle(join(save_path, file_name))
else:
    cca_df = pd.DataFrame(columns=['region_pair', 'eid'])

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
            clusters_pass[probe] = np.unique(spikes[probe].clusters)
        clusters_pass[probe] = clusters_pass[probe][~np.isin(clusters_pass[probe], artifact_neurons.loc[
            artifact_neurons['pid'] == pid, 'neuron_id'].values)]
        clusters[probe]['region'] = remap(clusters[probe]['acronym'], combine=True, abbreviate=True)

    # Create population activity arrays for all regions
    pca_opto, spks_opto = dict(), dict()
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

                # Subtract mean PSTH from each opto stim
                for tt in range(binned_spks_opto.shape[0]):
                    binned_spks_opto[tt, :, :] = binned_spks_opto[tt, :, :] - psth_opto['means']

                # Add to dict
                spks_opto[region] = binned_spks_opto

                # Perform PCA
                pca_opto[region] = np.empty([binned_spks_opto.shape[0], N_PC, binned_spks_opto.shape[2]])
                for tb in range(binned_spks_opto.shape[2]):
                    pca_opto[region][:, :, tb] = pca.fit_transform(binned_spks_opto[:, :, tb])


    # Perform CCA per region pair
    print('Starting CCA per region pair')
    for r, reg_pair in enumerate(REGION_PAIRS):
        region_1 = reg_pair[0]
        region_2 = reg_pair[1]

        # Skip if already processed
        if cca_df[(cca_df['region_pair'] == f'{region_1}-{region_2}') & (cca_df['eid'] == eid)].shape[0] > 0:
            print(f'Found {region_1}-{region_2} for {subject} {date}')
            continue

        if (region_1 in pca_opto.keys()) & (region_2 in pca_opto.keys()):
            print(f'Calculating {region_1}-{region_2}')

            # Fit to odd trials and project even trials
            r_opto = np.empty((2, psth_opto['tscale'].shape[0], N_MODES))
            reg_1_odd = np.concatenate(pca_opto[region_1][1::2, :, (psth_opto['tscale'] >= FIT_WIN[0])
                                                           & (psth_opto['tscale'] <= FIT_WIN[1])], axis=1).T
            reg_2_odd = np.concatenate(pca_opto[region_2][1::2, :, (psth_opto['tscale'] >= FIT_WIN[0])
                                                           & (psth_opto['tscale'] <= FIT_WIN[1])], axis=1).T
            cca.fit(reg_1_odd, reg_2_odd)
            for it, tb in enumerate(psth_opto['tscale']):
                x, y = cca.transform(pca_opto[region_1][::2, :, it], pca_opto[region_2][::2, :, it])
                for im in range(N_MODES):
                    r_opto[0, it, im] = pearsonr(x.T[im, :], y.T[im, :])[0]

            # Fit to even trials and project odd trials
            reg_1_even = np.concatenate(pca_opto[region_1][::2, :, (psth_opto['tscale'] >= FIT_WIN[0])
                                                           & (psth_opto['tscale'] <= FIT_WIN[1])], axis=1).T
            reg_2_even = np.concatenate(pca_opto[region_2][::2, :, (psth_opto['tscale'] >= FIT_WIN[0])
                                                           & (psth_opto['tscale'] <= FIT_WIN[1])], axis=1).T
            cca.fit(reg_1_even, reg_2_even)
            for it, tb in enumerate(psth_opto['tscale']):
                x, y = cca.transform(pca_opto[region_1][1::2, :, it], pca_opto[region_2][1::2, :, it])
                for im in range(N_MODES):
                    r_opto[1, it, im] = pearsonr(x.T[im, :], y.T[im, :])[0]

            # Take mean of the two folds (odd and even)
            r_opto = np.mean(r_opto, axis=0)

            # Add to dataframe
            cca_df = pd.concat((cca_df, pd.DataFrame(index=[cca_df.shape[0]], data={
                'subject': subject, 'date': date, 'eid': eid, 'region_1': region_1, 'region_2': region_2,
                'region_pair': f'{region_1}-{region_2}', 'r_opto': [r_opto],
                'n_modes': N_MODES, 'time': [psth_opto['tscale']], 'fit_window': [FIT_WIN]})))
    cca_df.to_pickle(join(save_path, file_name))

cca_df.to_pickle(join(save_path, file_name))
