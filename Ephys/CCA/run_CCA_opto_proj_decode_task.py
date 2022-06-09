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
from serotonin_functions import (paths, remap, query_ephys_sessions, load_trials,
                                 get_artifact_neurons, calculate_peths, load_passive_opto_times)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
N_MODES = 10  # number of CCA modes
MIN_TRIALS = 50  # minimum number of trials for both stim and non-stim conditions
OVERWRITE = True  # whether to overwrite existing runs
NEURON_QC = True  # whether to use neuron qc to exclude bad units
MIN_NEURONS = 10  # minimum neurons per region
WIN_SIZE = 0.05  # window size in seconds
PRE_TIME = 1.5  # time before stim onset in s
POST_TIME = 3.5  # time after stim onset in s
SMOOTHING = 0.1  # smoothing of psth
MAX_DELAY = 0.5  # max delay shift
SUBTRACT_MEAN = True  # whether to subtract the mean PSTH from each trial
MIN_FR = 0.5  # minimum firing rate over the whole recording
N_PC = 10  # number of PCs to use
FIT_TIME = 1.2  # timewindow relative to opto stim to use to fit CCA axes

# Initialize
cca = CCA(n_components=N_MODES, max_iter=1000)
pca = PCA(n_components=10)

# Paths
fig_path, save_path = paths()

# Initialize some things
REGION_PAIRS = [['M2', 'mPFC'], ['M2', 'ORB'], ['mPFC', 'Amyg'], ['ORB', 'Amyg'], ['M2', 'Amyg'],
                ['Hipp', 'PPC'], ['Hipp', 'Thal'], ['ORB', 'mPFC'], ['PPC', 'Thal'], ['MRN', 'SC'],
                ['RSP', 'SC'], ['BC', 'Str'], ['MRN', 'RSP'], ['MRN', 'SN'], ['Pir', 'Str'],
                ['SC', 'SN']]
np.random.seed(42)  # fix random seed for reproducibility
n_time_bins = int((PRE_TIME + POST_TIME) / WIN_SIZE)
if SMOOTHING > 0:
    w = n_time_bins - 1 if n_time_bins % 2 == 0 else n_time_bins
    window = gaussian(w, std=SMOOTHING / WIN_SIZE)
    window /= np.sum(window)

# Query sessions with behavior
rec = query_ephys_sessions(selection='aligned-behavior', one=one)

# Load in artifact neurons
artifact_neurons = get_artifact_neurons()

if ~OVERWRITE & isfile(join(save_path, f'cca_stimon_results_{WIN_SIZE}_binsize.pkl')):
    cca_df = pd.read_pickle(join(save_path, f'cca_stimon_results_{WIN_SIZE}_binsize.pkl'))
else:
    cca_df = pd.DataFrame(columns=['region_pair', 'eid'])

for i, eid in enumerate(np.unique(rec['eid'])):

    # Get session details
    subject = rec.loc[rec['eid'] == eid, 'subject'].values[0]
    date = rec.loc[rec['eid'] == eid, 'date'].values[0]
    print(f'Starting {subject}, {date}')

    # Load in trials
    trials = load_trials(eid, laser_stimulation=True)
    if ((trials[trials['laser_stimulation'] == 0].shape[0] < MIN_TRIALS)
            or (trials[trials['laser_stimulation'] == 0].shape[0] < MIN_TRIALS)):
        continue
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
            clusters_pass[probe] = np.unique(spikes.clusters)
        clusters_pass[probe] = clusters_pass[probe][~np.isin(clusters_pass[probe], artifact_neurons.loc[
            artifact_neurons['pid'] == pid, 'neuron_id'].values)]
        clusters[probe]['region'] = remap(clusters[probe]['acronym'], combine=True, abbreviate=True)

    # Create population activity arrays for all regions
    pca_opto, pca_no_opto = dict(), dict()
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
                fit_tb = np.argmin(np.abs(FIT_TIME - psth_opto['tscale']))
                pca_fit[region] = pca.fit(binned_spks_opto[:, :, fit_tb])
                pca_opto[region][:, :] = pca_fit[region][fit_tb].transform(binned_spks_opto[:, :, fit_tb])


                pca_opto[region] = np.empty([binned_spks_opto.shape[0], N_PC, binned_spks_opto.shape[2]])
                pca_fit[region] = np.empty(binned_spks_opto.shape[2]).astype(object)
                for tb in range(binned_spks_opto.shape[2]):
                    pca_fit[region][tb] = pca.fit(binned_spks_opto[:, :, tb])
                    pca_opto[region][:, :, tb] = pca_fit[region][tb].transform(binned_spks_opto[:, :, tb])




                # Get PSTH and binned spikes
                psth_opto, binned_spks_opto = calculate_peths(
                    spks_region, clus_region, np.unique(clus_region),
                    trials.loc[trials['laser_stimulation'] == 1, 'stimOn_times'], pre_time=PRE_TIME,
                    post_time=POST_TIME, bin_size=WIN_SIZE, smoothing=SMOOTHING, return_fr=False)
                psth_no_opto, binned_spks_no_opto = calculate_peths(
                    spks_region, clus_region, np.unique(clus_region),
                    trials.loc[trials['laser_stimulation'] == 0, 'stimOn_times'], pre_time=PRE_TIME,
                    post_time=POST_TIME, bin_size=WIN_SIZE, smoothing=SMOOTHING, return_fr=False)

                if SUBTRACT_MEAN:
                    # Subtract mean PSTH from each opto stim
                    for tt in range(binned_spks_opto.shape[0]):
                        binned_spks_opto[tt, :, :] = binned_spks_opto[tt, :, :] - psth_opto['means']
                    for tt in range(binned_spks_no_opto.shape[0]):
                        binned_spks_no_opto[tt, :, :] = binned_spks_no_opto[tt, :, :] - psth_no_opto['means']

                # Perform PCA
                pca_opto[region] = np.empty([binned_spks_opto.shape[0], N_PC, binned_spks_opto.shape[2]])
                for tb in range(binned_spks_opto.shape[2]):
                    pca_opto[region][:, :, tb] = pca.fit_transform(binned_spks_opto[:, :, tb])
                pca_no_opto[region] = np.empty([binned_spks_no_opto.shape[0], N_PC, binned_spks_no_opto.shape[2]])
                for tb in range(binned_spks_no_opto.shape[2]):
                    pca_no_opto[region][:, :, tb] = pca.fit_transform(binned_spks_no_opto[:, :, tb])

    # Perform CCA per region pair
    print('Starting CCA per region pair')
    for r1, region_1 in enumerate(pca_opto.keys()):
        for r2, region_2 in enumerate(list(pca_opto.keys())[r1:]):
            if region_1 == region_2:
                continue

            # Run CCA per region pair
            print(f'{region_1}-{region_2}')
            r_opto, r_no_opto = np.empty((N_MODES, n_time_bins)), np.empty((N_MODES, n_time_bins))

            for tb in range(n_time_bins):
                opto_x = np.empty(pca_opto[region_1][:, :, tb].shape[0])
                opto_y = np.empty(pca_opto[region_1][:, :, tb].shape[0])
                r_splits_opto = np.empty((2, N_MODES))
                r_splits_no_opto = np.empty((2, N_MODES))

                # Get odd and even trial indices
                even_ind_opto = np.arange(0, pca_opto[region_1][:, :, 0].shape[0], 2).astype(int)
                odd_ind_opto = np.arange(1, pca_opto[region_1][:, :, 0].shape[0], 2).astype(int)
                even_ind_no_opto = np.arange(0, pca_no_opto[region_1][:, :, 0].shape[0], 2).astype(int)
                odd_ind_no_opto = np.arange(1, pca_no_opto[region_1][:, :, 0].shape[0], 2).astype(int)

                # Fit on the even trials and correlate the odd trials
                cca.fit(pca_opto[region_1][even_ind_opto, :, tb], pca_opto[region_2][even_ind_opto, :, tb])
                x_opto, y_opto = cca.transform(pca_opto[region_1][odd_ind_opto, :, tb],
                                               pca_opto[region_2][odd_ind_opto, :, tb])

                cca.fit(pca_no_opto[region_1][even_ind_no_opto, :, tb],
                        pca_no_opto[region_2][even_ind_no_opto, :, tb])
                x_no_opto, y_no_opto = cca.transform(pca_no_opto[region_1][odd_ind_no_opto, :, tb],
                                                     pca_no_opto[region_2][odd_ind_no_opto, :, tb])

                # Get correlation and activity per mode
                for mm in range(N_MODES):
                    r_splits_opto[0, mm] = pearsonr(x_opto[:, mm], y_opto[:, mm])[0]  # correlate
                    r_splits_no_opto[0, mm] = pearsonr(x_no_opto[:, mm], y_no_opto[:, mm])[0]  # correlate

                # Fit on the odd trials and correlate the even trials
                cca.fit(pca_opto[region_1][odd_ind_opto, :, tb], pca_opto[region_2][odd_ind_opto, :, tb])
                x_opto, y_opto = cca.transform(pca_opto[region_1][even_ind_opto, :, tb],
                                               pca_opto[region_2][even_ind_opto, :, tb])

                cca.fit(pca_no_opto[region_1][odd_ind_no_opto, :, tb], pca_no_opto[region_2][odd_ind_no_opto, :, tb])
                x_no_opto, y_no_opto = cca.transform(pca_no_opto[region_1][even_ind_no_opto, :, tb],
                                               pca_no_opto[region_2][even_ind_no_opto, :, tb])

                # Correlate per mode
                for mm in range(N_MODES):
                    r_splits_opto[1, mm] = pearsonr(x_opto[:, mm], y_opto[:, mm])[0]
                    r_splits_no_opto[1, mm] = pearsonr(x_no_opto[:, mm], y_no_opto[:, mm])[0]

                # Get mean over splits
                r_opto[:, tb] = np.mean(r_splits_opto, axis=0).T
                r_no_opto[:, tb] = np.mean(r_splits_no_opto, axis=0).T

            # Baseline subtract
            r_baseline_opto, r_baseline_no_opto = r_opto.copy(), r_no_opto.copy()
            for mm in range(N_MODES):
                r_baseline_opto[mm, :] = r_opto[mm, :] - np.mean(r_opto[mm, psth_opto['tscale'] < 0])
                r_baseline_no_opto[mm, :] = r_no_opto[mm, :] - np.mean(r_no_opto[mm, psth_no_opto['tscale'] < 0])

            # Add to dataframe
            cca_df = pd.concat((cca_df, pd.DataFrame(index=[cca_df.shape[0]+1], data={
                'subject': subject, 'date': date, 'eid': eid, 'region_1': region_1, 'region_2': region_2,
                'region_pair': f'{region_1}-{region_2}', 'r_opto': [r_opto], 'r_baseline_opto': [r_baseline_opto],
                'r_no_opto': [r_no_opto], 'r_baseline_no_opto': [r_baseline_no_opto],
                'time': [psth_opto['tscale']]})), ignore_index=True)

        # Save results
        cca_df.to_pickle(join(save_path, f'cca_stimon_results_{WIN_SIZE}_binsize.pkl'))
