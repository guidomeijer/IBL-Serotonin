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
                                 get_artifact_neurons, calculate_peths, get_neuron_qc)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()
cca = CCA(n_components=1, max_iter=5000)
pca = PCA(n_components=10)

# Settings
OVERWRITE = True  # whether to overwrite existing runs
NEURON_QC = True  # whether to use neuron qc to exclude bad units
MIN_NEURONS = 10  # minimum neurons per region
WIN_SIZE = 0.05  # window size in seconds
PRE_TIME = 1.25  # time before stim onset in s
POST_TIME = 3.25  # time after stim onset in s
SMOOTHING = 0.1  # smoothing of psth
SUBTRACT_MEAN = False  # whether to subtract the mean PSTH from each trial
DIV_BASELINE = True  # whether to divide over baseline + 1 spk/s
K_FOLD = 5  # k in k-fold
K_FOLD_BOOTSTRAPS = 100  # how often to repeat the random trial selection
MIN_FR = 0.5  # minimum firing rate over the whole recording
N_PC = 10  # number of PCs to use

# Paths
fig_path, save_path = paths()

# Initialize some things
#REGION_PAIRS = [['M2', 'mPFC'], ['M2', 'ORB'], ['mPFC', 'Amyg'], ['ORB', 'Amyg'], ['M2', 'Amyg'],
#                ['Hipp', 'PPC'], ['Hipp', 'Thal'], ['ORB', 'mPFC'], ['PPC', 'Thal']]
REGION_PAIRS = [['M2', 'mPFC'], ['M2', 'OFC']]
np.random.seed(42)  # fix random seed for reproducibility
n_time_bins = int((PRE_TIME + POST_TIME) / WIN_SIZE)
kfold = KFold(n_splits=K_FOLD, shuffle=True)
if SMOOTHING > 0:
    w = n_time_bins - 1 if n_time_bins % 2 == 0 else n_time_bins
    window = gaussian(w, std=SMOOTHING / WIN_SIZE)
    window /= np.sum(window)

# Query sessions with frontal and amygdala
rec = query_ephys_sessions(one=one, acronym='MOs')

# Load in artifact neurons
artifact_neurons = get_artifact_neurons()

if ~OVERWRITE & isfile(join(save_path, 'cca_results.pickle')):
    cca_df = pd.read_pickle(join(save_path, 'cca_results.pickle'))
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
            qc_metrics = get_neuron_qc(pid, one=one, ba=ba)
            clusters_pass[probe] = np.where(qc_metrics['label'] == 1)[0]
        else:
            clusters_pass[probe] = np.unique(spikes.clusters)
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

                if DIV_BASELINE:
                    # Divide each trial over baseline + 1 spks/s
                    for nn in range(binned_spks_opto.shape[1]):
                        for tt in range(binned_spks_opto.shape[0]):
                            binned_spks_opto[tt, nn, :] = (binned_spks_opto[tt, nn, :]
                                                          / (np.mean(psth_opto['means'][nn, psth_opto['tscale'] < 0])
                                                             + (1/PRE_TIME)))

                if SUBTRACT_MEAN:
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
    all_cca_df = pd.DataFrame()
    for r, reg_pair in enumerate(REGION_PAIRS):
        region_1 = reg_pair[0]
        region_2 = reg_pair[1]

        # Skip if already processed
        if cca_df[(cca_df['region_pair'] == f'{region_1}-{region_2}') & (cca_df['eid'] == eid)].shape[0] > 0:
            print(f'Found {region_1}-{region_2} for {subject} {date}')
            continue

        if (region_1 in pca_opto.keys()) & (region_2 in pca_opto.keys()):
            print(f'Calculating {region_1}-{region_2}')

            # Run CCA
            r_mean, r_std, r_median = np.empty(n_time_bins), np.empty(n_time_bins), np.empty(n_time_bins)
            for tb in range(n_time_bins):
                if np.mod(tb, 20) == 0:
                    print(f'Timebin {tb} of {n_time_bins}..')
                opto_x = np.empty(pca_opto[region_1][:, :, 0].shape[0])
                opto_y = np.empty(pca_opto[region_1][:, :, 0].shape[0])

                r_boot = []
                for kk in range(K_FOLD_BOOTSTRAPS):
                    x_test, y_test = np.empty(opto_train_times.shape[0]), np.empty(opto_train_times.shape[0])
                    r_splits = []
                    for train_index, test_index in kfold.split(pca_opto[region_1][:, :, 0]):
                        cca.fit(pca_opto[region_1][train_index, :, tb],
                                pca_opto[region_2][train_index, :, tb])
                        x, y = cca.transform(pca_opto[region_1][test_index, :, tb],
                                             pca_opto[region_2][test_index, :, tb])
                        x_test[test_index] = x.T[0]
                        y_test[test_index] = y.T[0]
                        r_splits.append(pearsonr(x.T[0], y.T[0])[0])
                    r_boot.append(pearsonr(x_test, y_test)[0])
                    #r_boot.append(np.mean(r_splits))
                r_mean[tb] = np.mean(r_boot)
                r_median[tb] = np.median(r_boot)
                r_std[tb] = np.std(r_boot)

            # Add to dataframe
            cca_df = pd.concat((cca_df, pd.DataFrame(index=[cca_df.shape[0]], data={
                'subject': subject, 'date': date, 'eid': eid, 'region_1': region_1, 'region_2': region_2,
                'region_pair': f'{region_1}-{region_2}', 'r_mean': [r_mean], 'r_std': [r_std],
                'r_median': [r_median], 'time': [psth_opto['tscale']]})))

    # Save result
    cca_df.to_pickle(join(save_path, 'cca_results.pickle'))
    print('Results saved to disk')

# Save result
cca_df.to_pickle(join(save_path, 'cca_results.pickle'))
print('Done!')
