#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join
import pandas as pd
from brainbox.task.closed_loop import roc_single_event
from brainbox.metrics.single_units import spike_sorting_metrics
from brainbox.io.one import SpikeSortingLoader
from serotonin_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                                 remove_artifact_neurons)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
OVERWRITE = True
NEURON_QC = True
PRE_TIME = [0.5, 0]  # for significance testing
POST_TIME_EARLY = [0, 0.5]
POST_TIME_LATE = [0.5, 1]
BIN_SIZE = 0.05
PERMUTATIONS = 500
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'SingleNeurons', 'LightModNeurons')

# Query sessions
rec = query_ephys_sessions(one=one)

if OVERWRITE:
    light_neurons = pd.DataFrame()
else:
    light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
    rec = rec[~rec['eid'].isin(light_neurons['eid'])]

for i in rec.index.values:

    # Get session details
    pid, eid, probe = rec.loc[i, 'pid'], rec.loc[i, 'eid'], rec.loc[i, 'probe']
    subject, date = rec.loc[i, 'subject'], rec.loc[i, 'date']

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

    # Load in spikes
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    if 'acronym' not in clusters.keys():
        print(f'No brain regions found for {eid}')
        continue

    # Filter neurons that pass QC
    if NEURON_QC:
        print('Calculating neuron QC metrics..')
        qc_metrics, _ = spike_sorting_metrics(spikes.times, spikes.clusters,
                                              spikes.amps, spikes.depths,
                                              cluster_ids=np.arange(clusters.channels.size))
        clusters_pass = np.where(qc_metrics['label'] == 1)[0]
    else:
        clusters_pass = np.unique(spikes.clusters)
    spikes.times = spikes.times[np.isin(spikes.clusters, clusters_pass)]
    spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters_pass)]
    if len(spikes.clusters) == 0:
        continue

    # Select spikes of passive period
    start_passive = opto_train_times[0] - 360
    spikes.clusters = spikes.clusters[spikes.times > start_passive]
    spikes.times = spikes.times[spikes.times > start_passive]

    # Determine significant neurons
    print('Calculating modulation index for EARLY stim phase..')
    roc_auc, cluster_ids = roc_single_event(spikes.times, spikes.clusters,
                                            opto_train_times, pre_time=PRE_TIME,
                                            post_time=POST_TIME_EARLY)
    mod_idx_early = 2 * (roc_auc - 0.5)

    mod_idx_early_permut = np.zeros([PERMUTATIONS, len(np.unique(spikes.clusters))])
    for k in range(PERMUTATIONS):
        this_roc_auc_permut = roc_single_event(
            spikes.times, spikes.clusters,
            np.random.uniform(low=start_passive, high=opto_train_times[-1],
                              size=opto_train_times.shape[0]),
            pre_time=PRE_TIME, post_time=POST_TIME_EARLY)[0]
        mod_idx_early_permut[k, :] = 2 * (this_roc_auc_permut - 0.5)

    mod_early = ((mod_idx_early > np.percentile(mod_idx_early_permut, 97.5, axis=0))
                   | (mod_idx_early < np.percentile(mod_idx_early_permut, 2.5, axis=0)))
    enh_early = mod_idx_early > np.percentile(mod_idx_early_permut, 97.5, axis=0)
    supp_early = mod_idx_early < np.percentile(mod_idx_early_permut, 2.5, axis=0)

    print('Calculating modulation index for LATE stim phase..')
    roc_auc, cluster_ids = roc_single_event(spikes.times, spikes.clusters,
                                            opto_train_times, pre_time=PRE_TIME,
                                            post_time=POST_TIME_LATE)
    mod_idx_late = 2 * (roc_auc - 0.5)

    mod_idx_late_permut = np.zeros([PERMUTATIONS, len(np.unique(spikes.clusters))])
    for k in range(PERMUTATIONS):
        this_roc_auc_permut = roc_single_event(
            spikes.times, spikes.clusters,
            np.random.uniform(low=start_passive, high=opto_train_times[-1],
                              size=opto_train_times.shape[0]),
            pre_time=PRE_TIME, post_time=POST_TIME_LATE)[0]
        mod_idx_late_permut[k, :] = 2 * (this_roc_auc_permut - 0.5)

    mod_late = ((mod_idx_late > np.percentile(mod_idx_late_permut, 97.5, axis=0))
                   | (mod_idx_late < np.percentile(mod_idx_late_permut, 2.5, axis=0)))
    enh_late = mod_idx_late > np.percentile(mod_idx_late_permut, 97.5, axis=0)
    supp_late = mod_idx_late < np.percentile(mod_idx_late_permut, 2.5, axis=0)

    cluster_regions = remap(clusters.atlas_id[cluster_ids])
    light_neurons = pd.concat((light_neurons, pd.DataFrame(data={
        'subject': subject, 'date': date, 'eid': eid, 'probe': probe, 'pid': pid,
        'region': cluster_regions, 'neuron_id': cluster_ids,
        'mod_index_early': mod_idx_early, 'mod_index_late': mod_idx_late,
        'mod_null_early': np.mean(mod_idx_early_permut, axis=0),
        'mod_null_late': np.mean(mod_idx_late_permut, axis=0),
        'modulated_early': mod_early, 'enhanced_early': enh_early, 'suppressed_early': supp_early,
        'modulated_late': mod_late, 'enhanced_late': enh_late, 'suppressed_late': supp_late,
        'modulated': (mod_early | mod_late)})))

# Remove artifact neurons
light_neurons = remove_artifact_neurons(light_neurons)

# Save output
light_neurons.to_csv(join(save_path, 'light_modulated_neurons.csv'), index=False)
