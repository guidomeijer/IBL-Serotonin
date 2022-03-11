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
from sklearn.cross_decomposition import CCA
from brainbox.metrics.single_units import spike_sorting_metrics
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.io.one import SpikeSortingLoader
from serotonin_functions import (paths, remap, query_ephys_sessions, load_passive_opto_times,
                                 get_artifact_neurons)
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()
cca = CCA(n_components=1)

# Settings
NEURON_QC = True
MIN_NEURONS = 10  # per region
PULSE_TIME = 1
fig_path, save_path = paths()
fig_path = join(fig_path, 'Ephys', 'CCA')

# Query sessions
rec = query_ephys_sessions(one=one)

# Load in artifact neurons
artifact_neurons = get_artifact_neurons()

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
    spikes, clusters, channels = dict(), dict(), dict()
    for (pid, probe) in zip(rec.loc[rec['eid'] == eid, 'pid'].values, rec.loc[rec['eid'] == eid, 'probe'].values):

        sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
        spikes[probe], clusters[probe], channels[probe] = sl.load_spike_sorting()
        clusters[probe] = sl.merge_clusters(spikes[probe], clusters[probe], channels[probe])

        # Filter neurons that pass QC and artifact neurons
        if NEURON_QC:
            print('Calculating neuron QC metrics..')
            qc_metrics, _ = spike_sorting_metrics(spikes[probe].times, spikes[probe].clusters,
                                                  spikes[probe].amps, spikes[probe].depths,
                                                  cluster_ids=np.arange(clusters[probe].channels.size))
            clusters_pass = np.where(qc_metrics['label'] == 1)[0]
        else:
            clusters_pass = np.unique(spikes.clusters)
        clusters_pass = clusters_pass[~np.isin(clusters_pass, artifact_neurons.loc[
            artifact_neurons['pid'] == pid, 'neuron_id'].values)]
        spikes[probe].times = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_pass)]
        spikes[probe].clusters = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_pass)]
        clusters[probe]['region'] = remap(clusters[probe]['atlas_id'], combine=True)


    # Get time intervals for opto and spontaneous activity
    np.random.seed(42)
    spont_on_times = np.sort(np.random.uniform(opto_train_times[0] - (6 * 60), opto_train_times[0],
                                               size=opto_train_times.size))
    spont_times = np.column_stack((spont_on_times, spont_on_times + PULSE_TIME))
    opto_times = np.column_stack((opto_train_times, opto_train_times + PULSE_TIME))

    # Create population activity arrays for all regions
    pop_act = dict()
    for probe in spikes.keys():
        for region in np.unique(clusters[probe]['region']):

             # Select spikes and clusters in this brain region
             clusters_in_region = clusters[probe]['region'] == region
             spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)]
             clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_in_region)]
             #if len(clusters_in_region) < MIN_NEURONS:
             #    continue

             pop_act[region], _ = get_spike_counts_in_bins(spks_region, clus_region, spont_times)
             pop_act[region] = pop_act.T






