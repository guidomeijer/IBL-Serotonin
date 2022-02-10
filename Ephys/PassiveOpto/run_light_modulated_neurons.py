#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:17:26 2021
By: Guido Meijer
"""

import numpy as np
from os.path import join, isdir
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from os import mkdir
from brainbox.task.closed_loop import roc_single_event
from brainbox.metrics.single_units import spike_sorting_metrics
from serotonin_functions import figure_style
import brainbox.io.one as bbone
from brainbox.plot import peri_event_time_histogram
from serotonin_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                                 get_artifact_neurons, remove_artifact_neurons)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
OVERWRITE = True
NEURON_QC = True
T_BEFORE = 1  # for plotting
T_AFTER = 2
PRE_TIME = [0.5, 0]  # for significance testing
POST_TIME_EARLY = [0, 0.5]
POST_TIME_LATE = [0.5, 1]
BIN_SIZE = 0.05
PERMUTATIONS = 500
_, fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'SingleNeurons', 'LightModNeurons')

# Query sessions
eids, _, subjects = query_ephys_sessions(return_subjects=True, one=one)

if OVERWRITE:
    light_neurons = pd.DataFrame()
else:
    light_neurons = pd.read_csv(join(save_path, 'light_modulated_neurons.csv'))
    eids = eids[~np.isin(eids, light_neurons['eid'])]
for i, eid in enumerate(eids):

    # Get session details
    ses_details = one.get_details(eid)
    subject = ses_details['subject']
    date = ses_details['start_time'][:10]

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
    spikes, clusters, channels = bbone.load_spike_sorting_with_channel(
        eid, aligned=True, one=one, dataset_types=['spikes.amps', 'spikes.depths'], brain_atlas=ba)

    for p, probe in enumerate(spikes.keys()):
        if 'acronym' not in clusters[probe].keys():
            print(f'No brain regions found for {eid}')
            continue

        # Filter neurons that pass QC
        if NEURON_QC:
            print('Calculating neuron QC metrics..')
            qc_metrics, _ = spike_sorting_metrics(spikes[probe].times, spikes[probe].clusters,
                                                  spikes[probe].amps, spikes[probe].depths,
                                                  cluster_ids=np.arange(clusters[probe].channels.size))
            clusters_pass = np.where(qc_metrics['label'] == 1)[0]
        else:
            clusters_pass = np.unique(spikes[probe].clusters)
        spikes[probe].times = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_pass)]
        spikes[probe].clusters = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_pass)]
        if len(spikes[probe].clusters) == 0:
            continue

        # Select spikes of passive period
        start_passive = opto_train_times[0] - 360
        spikes[probe].clusters = spikes[probe].clusters[spikes[probe].times > start_passive]
        spikes[probe].times = spikes[probe].times[spikes[probe].times > start_passive]

        # Determine significant neurons
        print('Calculating modulation index for EARLY stim phase..')
        roc_auc, cluster_ids = roc_single_event(spikes[probe].times, spikes[probe].clusters,
                                                opto_train_times, pre_time=PRE_TIME,
                                                post_time=POST_TIME_EARLY)
        mod_idx_early = 2 * (roc_auc - 0.5)

        mod_idx_early_permut = np.zeros([PERMUTATIONS, len(np.unique(spikes[probe].clusters))])
        for k in range(PERMUTATIONS):
            this_roc_auc_permut = roc_single_event(
                spikes[probe].times, spikes[probe].clusters,
                np.random.uniform(low=start_passive, high=opto_train_times[-1],
                                  size=opto_train_times.shape[0]),
                pre_time=PRE_TIME, post_time=POST_TIME_EARLY)[0]
            mod_idx_early_permut[k, :] = 2 * (this_roc_auc_permut - 0.5)

        mod_early = ((mod_idx_early > np.percentile(mod_idx_early_permut, 97.5, axis=0))
                       | (mod_idx_early < np.percentile(mod_idx_early_permut, 2.5, axis=0)))
        enh_early = mod_idx_early > np.percentile(mod_idx_early_permut, 97.5, axis=0)
        supp_early = mod_idx_early < np.percentile(mod_idx_early_permut, 2.5, axis=0)

        print('Calculating modulation index for LATE stim phase..')
        roc_auc, cluster_ids = roc_single_event(spikes[probe].times, spikes[probe].clusters,
                                                opto_train_times, pre_time=PRE_TIME,
                                                post_time=POST_TIME_LATE)
        mod_idx_late = 2 * (roc_auc - 0.5)

        mod_idx_late_permut = np.zeros([PERMUTATIONS, len(np.unique(spikes[probe].clusters))])
        for k in range(PERMUTATIONS):
            this_roc_auc_permut = roc_single_event(
                spikes[probe].times, spikes[probe].clusters,
                np.random.uniform(low=start_passive, high=opto_train_times[-1],
                                  size=opto_train_times.shape[0]),
                pre_time=PRE_TIME, post_time=POST_TIME_LATE)[0]
            mod_idx_late_permut[k, :] = 2 * (this_roc_auc_permut - 0.5)

        mod_late = ((mod_idx_late > np.percentile(mod_idx_late_permut, 97.5, axis=0))
                       | (mod_idx_late < np.percentile(mod_idx_late_permut, 2.5, axis=0)))
        enh_late = mod_idx_late > np.percentile(mod_idx_late_permut, 97.5, axis=0)
        supp_late = mod_idx_late < np.percentile(mod_idx_late_permut, 2.5, axis=0)

        cluster_regions = remap(clusters[probe].atlas_id[cluster_ids])
        light_neurons = light_neurons.append(pd.DataFrame(data={
            'subject': subject, 'date': date, 'eid': eid, 'probe': probe,
            'region': cluster_regions, 'neuron_id': cluster_ids,
            'mod_index_early': mod_idx_early, 'mod_index_late': mod_idx_late,
            'mod_null_early': np.mean(mod_idx_early_permut, axis=0),
            'mod_null_late': np.mean(mod_idx_late_permut, axis=0),
            'modulated_early': mod_early, 'enhanced_early': enh_early, 'suppressed_early': supp_early,
            'modulated_late': mod_late, 'enhanced_late': enh_late, 'suppressed_late': supp_late,
            'modulated': (mod_early | mod_late)}))

# Remove artifact neurons
light_neurons = remove_artifact_neurons(light_neurons)

# Save output
light_neurons.to_csv(join(save_path, 'light_modulated_neurons.csv'), index=False)
