#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:20:14 2023
By: Guido Meijer
"""

import ssm
import numpy as np
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
from brainbox.io.one import SpikeSortingLoader
from sklearn.model_selection import KFold
from serotonin_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                                 get_artifact_neurons, get_neuron_qc, calculate_peths,
                                 high_level_regions)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
OVERWRITE = True
NEURON_QC = True
N_STATES = np.arange(2, 21)
PRE_TIME = 1
POST_TIME = 4
BIN_SIZE = 0.01
MIN_NEURONS = 5
K_FOLDS = 10
CV_SHUFFLE = True

# Get paths
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'SingleNeurons', 'LightModNeurons')
save_path = join(save_path, 'HMM')

# Query sessions
rec = query_ephys_sessions(anesthesia='no&both', one=one)

# Get light artifact units
artifact_neurons = get_artifact_neurons()

# Initialize k-fold cross validation
kf = KFold(n_splits=K_FOLDS, shuffle=CV_SHUFFLE, random_state=42)

if OVERWRITE:
    log_likelihood_df = pd.DataFrame()
else:
    log_likelihood_df = pd.read_csv(join(save_path, 'hmm_log_likelihood.csv'))
    rec = rec[~rec['pid'].isin(log_likelihood_df['pid'])]

for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']

    print(f'\nStarting {subject}, {date} ({i+1} of {len(rec)})')

    # Load in laser pulse times
    opto_train_times, _ = load_passive_opto_times(eid, one=one)
    if len(opto_train_times) == 0:
        print('Could not load light pulses')
        continue

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
    clusters['high_level_region'] = high_level_regions(clusters['acronym'])
    clusters_regions = clusters['high_level_region'][clusters_pass]

    # Loop over regions
    for r, region in enumerate(np.unique(clusters['high_level_region'])):

        if region == 'root':
            continue

        # Select spikes and clusters in this brain region
        clusters_in_region = clusters_pass[clusters_regions == region]
        if len(clusters_in_region) < MIN_NEURONS:
            continue

        # Get binned spikes centered at stimulation onset
        peth, binned_spikes = calculate_peths(spikes.times, spikes.clusters, clusters_in_region, opto_train_times,
                                              pre_time=PRE_TIME, post_time=POST_TIME, bin_size=BIN_SIZE,
                                              smoothing=0, return_fr=False)
        binned_spikes = binned_spikes.astype(int)
        time_ax = peth['tscale']

        # Create list of (time_bins x neurons) per stimulation trial
        trial_data = []
        for i in range(binned_spikes.shape[0]):
            trial_data.append(np.transpose(binned_spikes[i, :, :]))

        # Loop over different number of states
        log_likelihood = np.empty(N_STATES.shape[0])

        for j, s in enumerate(N_STATES):
            print(f'Starting state {s} of {N_STATES[-1]}')

            # Cross validate
            train_index, test_index = next(kf.split(trial_data))

            # Fit HMM on training data
            simple_hmm = ssm.HMM(s, clusters_in_region.shape[0], observations='poisson')
            lls = simple_hmm.fit(list(np.array(trial_data)[train_index]), method='em',
                                 transitions='sticky')

            # Get log-likelihood on test data
            log_likelihood[j] = simple_hmm.log_likelihood(list(np.array(trial_data)[test_index]))

        # Add to dataframe
        log_likelihood_df = pd.concat((log_likelihood_df, pd.DataFrame(data={
            'log_likelihood': log_likelihood, 'n_states': N_STATES, 'region': region,
            'subject': subject, 'pid': pid})))

    # Save result
    log_likelihood_df.to_csv(join(save_path, 'hmm_log_likelihood.csv'))




