#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 14:00:33 2021
By: Guido Meijer
"""

import pandas as pd
from os.path import join
import numpy as np
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import brainbox.io.one as bbone
from brainbox.task.closed_loop import generate_pseudo_blocks
from brainbox.population.decode import classify, get_spike_counts_in_bins
from serotonin_functions import (query_ephys_sessions, load_trials, paths, remap, load_subjects,
                                 behavioral_criterion)
from one.api import ONE
one = ONE()

# Settings
ARTIFACT_CUTOFF = 0.48
NEURON_QC = False
MIN_NEURONS = 5
PRE_TIME = 0
POST_TIME = 0.3
MIN_TRIALS = 300
ITERATIONS = 500
CROSS_VALIDATION = 'kfold'
N_SPLITS = 5
_, fig_path, save_path = paths()

# Load in data
eids, _, ses_subjects = query_ephys_sessions(one=one, return_subjects=True)
light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
subjects = load_subjects()

# Apply behavioral criterion
eids = behavioral_criterion(eids, min_trials=MIN_TRIALS)

# Initialize classifier
classifier = LDA()
cv = KFold(n_splits=N_SPLITS, shuffle=False)

# Loop over sessions
results_df = pd.DataFrame()
for i, eid in enumerate(eids):

    # Get session details
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]
    sert_cre = subjects.loc[subjects['subject'] == subject, 'sert-cre'].values[0]
    print(f'Starting {subject}, {date}')

    # Load in trials
    try:
        if sert_cre:
            trials = load_trials(eid, laser_stimulation=True)
        else:
            trials = load_trials(eid, patch_old_opto=False, laser_stimulation=True)
    except:
        print('Could not load trials')
        continue
    if trials.shape[0] < MIN_TRIALS:
        continue
    block_id = (trials['probabilityLeft'] == 0.8).astype(int).values

    # Load in neural data
    spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, aligned=True, one=one)

    for p, probe in enumerate(spikes.keys()):
        if 'acronym' not in clusters[probe].keys():
            print(f'No brain regions found for {eid}')
            continue

        # Filter neurons that pass QC
        if ('metrics' not in clusters[probe].keys()) or (NEURON_QC == False):
            print('No neuron QC, using all neurons')
            clusters_pass = np.unique(spikes[probe].clusters)
        else:
            clusters_pass = np.where(clusters[probe]['metrics']['label'] == 1)[0]
        spikes[probe].times = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_pass)]
        spikes[probe].clusters = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_pass)]
        if len(spikes[probe].clusters) == 0:
            continue

        # Exclude artifact neurons
        artifact_neurons = light_neurons.loc[(light_neurons['eid'] == eid) & (light_neurons['probe'] == probe)
                                             & (light_neurons['roc_auc'] > ARTIFACT_CUTOFF), 'cluster_id'].values
        spikes[probe].times = spikes[probe].times[~np.isin(spikes[probe].clusters, artifact_neurons)]
        spikes[probe].clusters = spikes[probe].clusters[~np.isin(spikes[probe].clusters, artifact_neurons)]
        print(f'Excluded {len(artifact_neurons)} light artifact neurons')

        # Remap to beryl atlas
        clusters[probe]['acronym'] = remap(clusters[probe]['atlas_id'])

        # Loop over regions
        for r, region in enumerate(np.unique(clusters[probe]['acronym'])):

            # Get clusters in this brain region
            clusters_in_region = np.where(clusters[probe]['acronym'] == region)[0]

            # Select spikes and clusters
            spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)]
            clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_in_region)]

            # Check if there are enough neurons in this brain region
            if np.unique(clus_region).shape[0] < MIN_NEURONS:
                continue
            print(f'Processing region {region} ({np.unique(clus_region).shape[0]} neurons)')

            # Get population activity for all trials
            times = np.column_stack(((trials['stimOn_times'] - PRE_TIME), (trials['stimOn_times'] + POST_TIME)))
            population_activity, cluster_ids = get_spike_counts_in_bins(spks_region, clus_region, times)
            population_activity = population_activity.T

            # Get block trials
            block_on_trials = (trials['laser_stimulation'] == 1) & (trials['laser_probability'] != 0.25)
            block_off_trials = (trials['laser_stimulation'] == 0) & (trials['laser_probability'] != 0.75)

            # Decode
            acc_block_on = classify(population_activity[block_on_trials, :], block_id[block_on_trials],
                                    classifier, cross_validation=cv)[0]
            acc_block_off = classify(population_activity[block_off_trials], block_id[block_off_trials],
                               classifier, cross_validation=cv)[0]
            acc_improvement = acc_block_on - acc_block_off

            # Get p-value for stim induced decoding change
            acc_shuf_block_on, acc_shuf_block_off = np.empty(ITERATIONS), np.empty(ITERATIONS)
            for k in range(ITERATIONS):
                pseudo_opto = generate_pseudo_blocks(trials.shape[0], first5050=0)
                pseudo_opto = (pseudo_opto == 0.8).astype(int)
                acc_shuf_block_on[k] = classify(population_activity[pseudo_opto == 1, :], block_id[pseudo_opto == 1],
                                                classifier, cross_validation=cv)[0]
                acc_shuf_block_off[k] = classify(population_activity[pseudo_opto == 0], block_id[pseudo_opto == 0],
                                                 classifier, cross_validation=cv)[0]
            acc_shuf_improvement = acc_shuf_block_on - acc_shuf_block_off
            if acc_improvement > 0:
                p_value = np.sum(acc_shuf_improvement > acc_improvement) / ITERATIONS
            else:
                p_value = np.sum(acc_shuf_improvement < acc_improvement) / ITERATIONS

            # Add to dataframe
            results_df = results_df.append(pd.DataFrame(
                  index=[results_df.shape[0] + 1], data={
                      'subject': subject, 'date': date, 'eid': eid, 'probe': probe, 'region': region,
                      'sert-cre': sert_cre, 'acc_block_on': acc_block_on, 'acc_block_off': acc_block_off,
                      'acc_improvement': acc_improvement, 'p_value': p_value}))

    results_df.to_csv(join(save_path, f'lda_decoding_block_{PRE_TIME}_{POST_TIME}.csv'))
results_df.to_csv(join(save_path, f'lda_decoding_block_{PRE_TIME}_{POST_TIME}.csv'))
